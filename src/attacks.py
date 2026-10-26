"""Step 4 - Attacks applied at EVALUATION time, after the watermarked image is published.

Unlike augmentations.py, nothing here needs gradients: these are the edits the image
undergoes "in the wild". Some were seen during marking (rotation, crop, resize, blur),
others were never seen (real JPEG compression, brightness), which tests generalization.
Every attacked image is brought back to 224x224 and quantized to 8 bits before detection.
"""

import io
import math

import torch
import torchvision.transforms.functional as TF
from PIL import Image

SIZE = 224


def jpeg(img, quality=50):
    """Real (non-differentiable) JPEG compression with PIL."""
    buffer = io.BytesIO()
    TF.to_pil_image(img).save(buffer, format="JPEG", quality=quality)
    return TF.to_tensor(Image.open(buffer))


def center_crop(img, area=0.5):
    """Keep the central 50% of the area, then resize back to 224x224."""
    side = int(SIZE * math.sqrt(area))
    return TF.resize(TF.center_crop(img, [side, side]), [SIZE, SIZE], antialias=True)


def resize(img, scale=0.5):
    """Downscale by 0.5 (e.g. a thumbnail), then upscale back to 224x224 for the encoder."""
    small = int(SIZE * scale)
    return TF.resize(TF.resize(img, [small, small], antialias=True), [SIZE, SIZE], antialias=True)


def rotation(img, angle=25):
    return TF.rotate(img, angle)


def blur(img, sigma=2.0):
    return TF.gaussian_blur(img, kernel_size=13, sigma=sigma)


def brightness(img, factor=1.5):
    return TF.adjust_brightness(img, factor)


ATTACKS = {
    "identity": lambda img: img,
    "jpeg_50": jpeg,
    "crop_50": center_crop,
    "resize_0.5": resize,
    "rotation_25": rotation,
    "blur_2": blur,
    "brightness_1.5": brightness,
}


def attack(img, name):
    """Apply the attack `name` to a (3, 224, 224) image and quantize the result to 8-bit pixels."""
    out = ATTACKS[name](img.cpu()).clamp(0, 1)
    return torch.round(out * 255) / 255
