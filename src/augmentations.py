"""Step 2 - Differentiable augmentations used DURING marking.

At every optimization step we feed a randomly transformed version of the watermarked image
to the encoder. Gradients must flow through the transformation back to the pixels, so every
operation here is a differentiable tensor op (interpolation, grid sampling, convolution).
This is what makes the mark robust: it is optimized to survive these transformations.
(Compare with attacks.py, which is applied AFTER publication and need not be differentiable.)
"""

import math
import random

import torchvision.transforms.functional as TF

SIZE = 224


def rotate(img):
    """Rotation by a random angle in [-30, 30] degrees."""
    return TF.rotate(img, random.uniform(-30, 30))


def crop(img):
    """Random square crop keeping 50-100% of the area, resized back to 224x224."""
    side = int(SIZE * math.sqrt(random.uniform(0.5, 1.0)))
    top, left = random.randint(0, SIZE - side), random.randint(0, SIZE - side)
    return TF.resized_crop(img, top, left, side, side, [SIZE, SIZE], antialias=True)


def resize(img):
    """Downscale by a random factor in [0.5, 1], then upscale back to 224x224 (loses details)."""
    small = int(SIZE * random.uniform(0.5, 1.0))
    return TF.resize(TF.resize(img, [small, small], antialias=True), [SIZE, SIZE], antialias=True)


def blur(img):
    """Gaussian blur with a random odd kernel size k in {3,...,9}, sigma = 0.15k + 0.35 (as in the paper)."""
    k = random.choice([3, 5, 7, 9])
    return TF.gaussian_blur(img, kernel_size=k, sigma=0.15 * k + 0.35)


def identity(img):
    return img


AUGMENTATIONS = [identity, rotate, crop, resize, blur, TF.hflip]


def random_augment(img):
    """Apply one augmentation picked uniformly at random."""
    return random.choice(AUGMENTATIONS)(img)
