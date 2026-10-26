"""Step 5 - Evaluation: detection, TPR per attack, false positive check, result table and figures."""

import csv
import os

import matplotlib
matplotlib.use("Agg")  # no display needed (works in scripts, on servers and in Colab)
import matplotlib.pyplot as plt
import torch

from src.attacks import ATTACKS, attack
from src.data import iterate_batches


@torch.no_grad()
def abs_cosines(feature_fn, imgs, key, batch_size=64):
    """|cos(phi(img), a)| for a batch of images (N, 3, 224, 224). Detection = this value > cos(theta)."""
    out = []
    for i in range(0, len(imgs), batch_size):
        x = feature_fn(imgs[i:i + batch_size].to(key.device))
        out.append(((x @ key).abs() / x.norm(dim=-1)).cpu())
    return torch.cat(out)


@torch.no_grad()
def negative_cosines(feature_fn, paths, key):
    """|cos| of unmarked images, loaded batch by batch (used for the false positive sanity check)."""
    return torch.cat([abs_cosines(feature_fn, b, key) for b in iterate_batches(paths)])


def attack_cosines(feature_fn, marked, key):
    """For each attack: attack every marked image, then compute its |cos| with the key."""
    return {name: abs_cosines(feature_fn, torch.stack([attack(img, name) for img in marked]), key)
            for name in ATTACKS}


def save_results(row, out_dir):
    """Add (or replace) this run's row in results.csv, and rewrite results.md for the README."""
    csv_path = os.path.join(out_dir, "results.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run"] != row["run"]]
    rows.append({k: str(v) for k, v in row.items()})

    columns = list(row.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "---|" * len(columns) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r.get(c, "") for c in columns) + " |\n")
    return rows


def plot_cos_hist(neg_cos, marked_cos, cos_theta, run, path):
    """Histogram of |cos(x, a)| for unmarked vs. marked images, with the detection threshold."""
    plt.figure(figsize=(6, 3.5))
    bins = torch.linspace(0, 1, 101).numpy()
    plt.hist(neg_cos.numpy(), bins=bins, alpha=0.7, density=True, label="unmarked")
    plt.hist(marked_cos.numpy(), bins=bins, alpha=0.7, density=True, label="marked (no attack)")
    plt.axvline(cos_theta, color="k", linestyle="--", label=f"threshold cos θ = {cos_theta:.3f}")
    plt.xlabel("|cos(x, a)|")
    plt.ylabel("density")
    plt.title(run)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_tpr(rows, path):
    """Grouped bar chart: TPR per attack, one bar per evaluated run."""
    attacks = list(ATTACKS)
    width = 0.8 / len(rows)
    plt.figure(figsize=(9, 3.8))
    for i, r in enumerate(rows):
        xs = [j + (i - (len(rows) - 1) / 2) * width for j in range(len(attacks))]
        plt.bar(xs, [float(r[a]) for a in attacks], width=width, label=r["run"])
    plt.xticks(range(len(attacks)), attacks, rotation=20)
    plt.ylabel("TPR @ FPR = 1e-6")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_examples(originals, marked, path, n=3):
    """Original / watermarked / amplified watermark (x10, centered on gray) for a few images."""
    n = min(n, len(marked))
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n), squeeze=False)
    for i in range(n):
        orig, wm = originals[i].cpu(), marked[i].cpu()
        diff = (0.5 + 10 * (wm - orig)).clamp(0, 1)
        for ax, im, title in zip(axes[i], [orig, wm, diff], ["original", "watermarked", "watermark x10"]):
            ax.imshow(im.permute(1, 2, 0).numpy())
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
