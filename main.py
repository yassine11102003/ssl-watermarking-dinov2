"""Run the full zero-bit watermarking experiment for one DINOv2 model.

    python main.py --model dinov2_vits14 --debug   # 5 images, quick sanity check
    python main.py --model dinov2_vits14
    python main.py --model dinov2_vits14_reg
    python main.py --model dinov2_vits14 --no_whitening
"""

import argparse
import os
import random

import torch
from tqdm import tqdm

from src.data import download_coco, load_image, make_splits
from src.encoder import fit_whitening, identity_whitening, load_dinov2, make_feature_fn
from src.evaluate import (attack_cosines, negative_cosines, abs_cosines, plot_cos_hist,
                          plot_tpr, save_examples, save_results)
from src.watermark import cos_threshold, make_key, mark_image, psnr


def parse_args():
    p = argparse.ArgumentParser(description="Watermarking images in DINOv2 latent spaces (zero-bit).")
    p.add_argument("--model", default="dinov2_vits14", help="dinov2_vits14 or dinov2_vits14_reg")
    p.add_argument("--n_mark", type=int, default=100, help="images to watermark")
    p.add_argument("--n_whiten", type=int, default=3900, help="unmarked images to fit PCA-whitening")
    p.add_argument("--n_negative", type=int, default=1000, help="unmarked images for the FPR check")
    p.add_argument("--psnr", type=float, default=40.0, help="target PSNR (dB)")
    p.add_argument("--fpr", type=float, default=1e-6, help="target false positive rate")
    p.add_argument("--iters", type=int, default=100, help="optimization steps per image")
    p.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    p.add_argument("--no_whitening", action="store_true", help="ablation: use raw DINOv2 features")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="5 marked images, small splits, results/debug/")
    args = p.parse_args()
    if args.debug:
        args.n_mark, args.n_whiten, args.n_negative = 5, 1000, 100
        args.out_dir = os.path.join(args.out_dir, "debug")
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    run = args.model + ("_nowhiten" if args.no_whitening else "")
    print(f"Run: {run} | device: {device}")

    # 1. Data: disjoint splits of COCO val2017
    img_dir = download_coco(args.data_dir)
    splits = make_splits(img_dir, args.n_whiten, args.n_mark, args.n_negative, args.seed)

    # 2. Feature extractor phi = whitening o DINOv2 (frozen)
    model = load_dinov2(args.model, device)
    d = model.embed_dim
    if args.no_whitening:
        mean, P = identity_whitening(d, device)
    else:
        print(f"Fitting PCA-whitening on {args.n_whiten} images ...")
        mean, P = fit_whitening(model, splits["whiten"], device)
    feature_fn = make_feature_fn(model, mean, P)

    # 3. Secret key and detection threshold
    key = make_key(d, args.seed).to(device)
    cos_theta = cos_threshold(args.fpr, d)
    print(f"d = {d}, FPR = {args.fpr:g} -> cos(theta) = {cos_theta:.4f}")

    # 4. Watermark the images
    originals, marked, psnrs = [], [], []
    for path in tqdm(splits["mark"], desc="Marking"):
        img = load_image(path).to(device)
        img_w = mark_image(img, feature_fn, key, cos_theta, args.psnr, args.iters, args.lr)
        originals.append(img.cpu())
        marked.append(img_w.cpu())
        psnrs.append(psnr(img_w, img))

    # 5. Evaluate: attacks on marked images, and false positives on unmarked ones
    print("Evaluating ...")
    cos_attacked = attack_cosines(feature_fn, marked, key)
    neg_cos = negative_cosines(feature_fn, splits["negative"], key)
    orig_cos = abs_cosines(feature_fn, torch.stack(originals), key)

    row = {"run": run}
    row.update({name: f"{(c > cos_theta).float().mean():.2f}" for name, c in cos_attacked.items()})
    row["psnr"] = f"{sum(psnrs) / len(psnrs):.2f}"
    row["false_pos"] = f"{int((neg_cos > cos_theta).sum())}/{len(neg_cos)}"
    row["cos_unmarked"] = f"{orig_cos.mean():.3f}"
    row["cos_marked"] = f"{cos_attacked['identity'].mean():.3f}"
    for k, v in row.items():
        print(f"  {k:>15}: {v}")

    # 6. Save table and figures
    rows = save_results(row, args.out_dir)
    plot_cos_hist(neg_cos, cos_attacked["identity"], cos_theta, run,
                  os.path.join(args.out_dir, f"cos_hist_{run}.png"))
    plot_tpr(rows, os.path.join(args.out_dir, "tpr_per_attack.png"))
    save_examples(originals, marked, os.path.join(args.out_dir, f"examples_{run}.png"))
    print(f"Saved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
