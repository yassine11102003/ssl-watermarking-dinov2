"""Step 3 - Zero-bit watermarking: secret key, detection threshold, perceptual constraints, marking loop."""

import torch
import torch.nn.functional as F
from scipy.special import betaincinv

from src.augmentations import random_augment


def make_key(d, seed=0):
    """Secret key: a random unit vector a in R^d (the same key for every image)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, generator=g)
    return a / a.norm()


def cos_threshold(fpr, d):
    """Half-angle of the dual hypercone D = {x : |cos(x, a)| > cos(theta)} for a target false positive rate.

    For a feature direction uniform on the sphere, P(|cos(x, a)| > c) = 1 - I_{c^2}(1/2, (d-1)/2),
    where I is the regularized incomplete Beta function. Inverting it gives c = cos(theta).
    """
    return float(betaincinv(0.5, (d - 1) / 2, 1 - fpr)) ** 0.5


def texture_map(img, k=7):
    """Perceptual mask H in [0, 1]: local pixel std over a k x k window, averaged over channels.

    Simplified version of the paper's SSIM attenuation: changes are less visible in textured
    areas than in flat ones (sky, walls), so we let the watermark live where H is large.
    """
    x = img.unsqueeze(0)                                   # (1, 3, H, W)
    mean = F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
    sq_mean = F.avg_pool2d(x ** 2, k, stride=1, padding=k // 2, count_include_pad=False)
    std = (sq_mean - mean ** 2).clamp(min=0).sqrt().mean(dim=1, keepdim=True)  # (1, 1, H, W)
    return (std / std.max().clamp(min=1e-8))[0]            # (1, H, W), broadcast over channels


def psnr(img_a, img_b):
    """PSNR in dB between two [0, 1] images."""
    mse = ((img_a - img_b) ** 2).mean()
    return (10 * torch.log10(1 / mse)).item()


def psnr_clip(w, target_psnr):
    """Rescale the watermark w if it is too strong, so that PSNR(img + w, img) >= target_psnr.

    PSNR = 10 log10(1 / MSE) for pixels in [0, 1], so the allowed energy is MSE_max = 10^(-PSNR/10).
    """
    mse_max = 10 ** (-target_psnr / 10)
    mse = (w ** 2).mean()
    if mse > mse_max:
        w = w * torch.sqrt(mse_max / mse)
    return w


def watermark_loss(x, key, cos_theta):
    """-[(x.a)^2 - ||x||^2 cos^2(theta)]: negative inside the cone, and lower the deeper we go.

    Minimizing it pushes the feature x towards the key axis, well inside the detection cone.
    """
    return -((x @ key) ** 2 - (x ** 2).sum(dim=-1) * cos_theta ** 2).mean()


def mark_image(img, feature_fn, key, cos_theta, target_psnr=40.0, iters=100, lr=0.01):
    """Watermark one (3, 224, 224) image by gradient descent on a pixel perturbation delta.

    At each step the watermark actually added is w = psnr_clip(H * delta): H hides it in
    textured regions and psnr_clip enforces the distortion budget. A random differentiable
    augmentation is applied before the encoder, so the mark is learned to survive it.
    """
    H = texture_map(img)
    delta = torch.zeros_like(img, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(iters):
        w = psnr_clip(H * delta, target_psnr)              # impose perceptual + PSNR constraints
        img_w = (img + w).clamp(0, 1)
        x = feature_fn(random_augment(img_w).unsqueeze(0))  # feature of an augmented view
        loss = watermark_loss(x, key, cos_theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        img_w = (img + psnr_clip(H * delta, target_psnr)).clamp(0, 1)
        return torch.round(img_w * 255) / 255              # a real image has integer pixel values
