# Watermarking images in DINOv2 latent spaces

A small, self-contained reproduction of the zero-bit method of
**Fernandez et al., *Watermarking Images in Self-Supervised Latent Spaces*, ICASSP 2022**
([arXiv:2112.09581](https://arxiv.org/abs/2112.09581), [official code](https://github.com/facebookresearch/ssl_watermarking)),
re-run with modern self-supervised encoders.

**Research question.** Are modern SSL encoders (DINOv2) good watermarking spaces, and do
*registers* (DINOv2-reg) change the robustness of the watermark?

## 1. Motivation

Classic watermarking hides a signal in a hand-crafted transform domain (DCT, DFT, wavelets).
Fernandez et al. replace the transform by a frozen self-supervised network: SSL features are
trained to be *invariant to data augmentation*, which is exactly the robustness a watermark needs,
and they come "for free", with no encoder/decoder trained for watermarking. The original paper
uses a ResNet-50 trained with DINO (d = 2048). This repo asks whether the idea transfers to
DINOv2 ViTs, whose features are much stronger for retrieval but which were never trained with
rotation augmentation, and whether the register tokens of DINOv2-reg, which clean up the
attention maps, also make the CLS token a better place to hide a mark.

## 2. Method (zero-bit)

```
                      secret key a (unit vector in R^d), threshold cos θ from the target FPR
                                               │
 image I ──► + w ──► random differentiable ──► DINOv2 (frozen) ──► PCA-whitening ──► x ──► loss(x, a)
   │          ▲      augmentation              CLS token, d=384                              │
   │          │                                                                              │
   │   w = psnr_clip(H ⊙ δ)  ◄────────────── Adam on δ (100 steps) ◄───── backprop to pixels ┘
   │          H = texture map of I
   ▼
 watermarked image  I_w = round(I + w)   ──► attacks (JPEG, crop, ...) ──► x ──► detect: |cos(x, a)| > cos θ
```

**Feature space.** φ(I) = whiten(DINOv2(normalize(I))), the CLS token of ViT-S/14 (d = 384)
followed by PCA-whitening learned on unmarked images, so that natural features are centered with
identity covariance and no direction of R^d is privileged.

**Detection region (dual hypercone).** With a secret unit key `a`:

$$\mathcal{D} = \{x \in \mathbb{R}^d : |x^\top a| > \|x\|\cos\theta\}$$

**Threshold from the false positive rate.** If a natural feature has a uniformly random direction,

$$\mathrm{FPR} = P(|\cos(x, a)| > \cos\theta) = 1 - I_{\cos^2\theta}\left(\tfrac12, \tfrac{d-1}{2}\right)
\;\Rightarrow\; \cos\theta = \sqrt{I^{-1}\left(\tfrac12, \tfrac{d-1}{2}, 1-\mathrm{FPR}\right)}$$

where I is the regularized incomplete Beta function (`scipy.special.betaincinv`).
For d = 384 and FPR = 1e-6 this gives **cos θ = 0.246** (it is 0.108 for d = 2048 in the paper:
lower dimension forces a narrower cone, i.e. a harder target).

**Marking loss** (the opposite of the "robustness estimate" of Cox et al.):

$$\mathcal{L}_w(x) = -\left[(x^\top a)^2 - \|x\|^2\cos^2\theta\right]$$

It is positive outside the cone and keeps decreasing as x moves deeper inside, which buys a margin
against attacks.

**Imperceptibility.** The perturbation actually added is `w = psnr_clip(H ⊙ δ, 40 dB)`.
`H` is a texture map (local 7×7 standard deviation, normalized to [0, 1]), a simplified version of
the paper's SSIM attenuation that hides the mark in textured regions; `psnr_clip` rescales `w` if
its PSNR drops below the target. The final image is rounded to 8-bit pixels.

**Robustness.** At each of the 100 Adam steps (lr = 0.01), the watermarked image goes through one
random *differentiable* augmentation (identity, rotation ±30°, crop 50–100 %, resize 0.5–1,
Gaussian blur, horizontal flip), so the feature is pushed into the cone for all these views.

## 3. Code organization

One file per step of the method, readable top to bottom:

| file | role |
|---|---|
| `src/data.py` | download COCO val2017, build disjoint splits, load images as 224×224 tensors in [0, 1] |
| `src/encoder.py` | frozen DINOv2 (CLS token), ImageNet normalization, PCA-whitening (fit + apply) |
| `src/augmentations.py` | **differentiable** augmentations used *during marking* (gradients flow through them) |
| `src/watermark.py` | secret key, threshold cos θ from the FPR, texture map, `psnr_clip`, loss, marking loop |
| `src/attacks.py` | attacks applied *at evaluation*, after publication (incl. real, non-differentiable JPEG) |
| `src/evaluate.py` | detection, TPR per attack, false positive check, results table and figures |
| `main.py` | runs the whole experiment for one model and appends a row to `results/results.csv` |

`augmentations.py` and `attacks.py` are deliberately separate: the first is *what we train the
mark against* (must be differentiable), the second is *what the image suffers in the wild* (no
gradients needed, and it includes edits never seen at marking time, like JPEG and brightness).

## 4. How to run

**Data.** COCO val2017 (5 000 images, ~800 MB) is downloaded automatically into `data/`.
Splits are disjoint and fixed by the seed: 3 900 images to fit the whitening, 100 images to
watermark, 1 000 unmarked images to check false positives.

### Locally

```bash
pip install -r requirements.txt
python main.py --model dinov2_vits14 --debug          # 5 images, sanity check (results/debug/)
python main.py --model dinov2_vits14
python main.py --model dinov2_vits14_reg
python main.py --model dinov2_vits14 --no_whitening   # ablation
```

### In Google Colab (free T4 GPU)

`Runtime > Change runtime type > T4 GPU`, then in a cell:

```python
!git clone https://github.com/yassine11102003/ssl-watermarking-dinov2.git
%cd ssl-watermarking-dinov2
!pip install -r requirements.txt
!python main.py --model dinov2_vits14 --debug
!python main.py --model dinov2_vits14
!python main.py --model dinov2_vits14_reg
!python main.py --model dinov2_vits14 --no_whitening
```

Each full run takes a few minutes on a T4 (≈ 50 min on a laptop CPU). The GPU is used automatically
when available. Main options: `--n_mark 100 --n_whiten 3900 --n_negative 1000 --psnr 40 --fpr 1e-6
--iters 100 --lr 0.01 --seed 0 --data_dir data --out_dir results`.

## 5. Results

RESULTS_PLACEHOLDER

## 7. Limitations

- **Marking is slow**: 100 forward/backward passes per image, versus a single forward pass for
  encoder/decoder methods like HiDDeN.
- **d = 384 vs. 2048** in the paper: a smaller space means a narrower detection cone
  (cos θ = 0.246 vs. 0.108 at FPR = 1e-6) and less room to hide the signal.
- **Small scale**: 100 marked images (TPR resolution 0.01) at 224×224 only; the FPR check over
  1 000 images can only show that the FPR is not grossly above 1e-3, not that it equals 1e-6.
- **Simplified perceptual mask**: local standard deviation instead of the paper's SSIM heatmap.
- **No security study**: an attacker who knows the method can estimate the key or run an
  adversarial attack on the public encoder to remove (or forge) the mark.
- Zero-bit only (no multi-bit message decoding).

## Reference

```bibtex
@inproceedings{fernandez2022watermarking,
  title     = {Watermarking Images in Self-Supervised Latent Spaces},
  author    = {Fernandez, Pierre and Sablayrolles, Alexandre and Furon, Teddy and J{\'e}gou, Herv{\'e} and Douze, Matthijs},
  booktitle = {ICASSP},
  year      = {2022}
}
```

DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
Registers: Darcet et al., *Vision Transformers Need Registers*, ICLR 2024.
