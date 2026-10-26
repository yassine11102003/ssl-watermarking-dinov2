"""Step 1 - Feature extractor phi: frozen DINOv2 (CLS token) followed by PCA-whitening."""

import torch
from torchvision import transforms

from src.data import iterate_batches

# DINOv2 expects ImageNet-normalized inputs; our images stay in [0, 1] everywhere else.
_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def load_dinov2(name, device):
    """Load a pre-trained DINOv2 from torch.hub (e.g. 'dinov2_vits14' or 'dinov2_vits14_reg') and freeze it."""
    model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)  # the network is never trained: only image pixels are optimized
    return model


def embed(model, imgs):
    """Raw DINOv2 feature: the CLS token (d = 384 for ViT-S/14) of a batch of [0, 1] images."""
    return model(_normalize(imgs))


@torch.no_grad()
def fit_whitening(model, paths, device, eps=1e-5):
    """Learn PCA-whitening on unmarked images: x -> (x - mean) @ P, with P = V / sqrt(eigenvalues + eps).

    After this map, the features are centered with (approximately) identity covariance,
    so a random secret direction is not "far" from where natural features live.
    """
    feats = torch.cat([embed(model, b.to(device)).cpu() for b in iterate_batches(paths)]).double()
    mean = feats.mean(dim=0)
    cov = torch.cov((feats - mean).T)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # cov = V diag(eigvals) V^T
    P = eigvecs / torch.sqrt(eigvals + eps)    # rotate onto principal axes, then rescale each axis
    return mean.float().to(device), P.float().to(device)


def identity_whitening(d, device):
    """'No whitening' ablation: zero mean and identity projection, i.e. raw DINOv2 features."""
    return torch.zeros(d, device=device), torch.eye(d, device=device)


def make_feature_fn(model, mean, P):
    """Return phi(img) = whiten(dinov2(img)), the feature space in which we watermark."""
    def feature_fn(imgs):
        return (embed(model, imgs) - mean) @ P
    return feature_fn
