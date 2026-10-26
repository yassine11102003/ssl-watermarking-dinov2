"""Step 0 - Data: download COCO val2017, build disjoint splits, load images as 224x224 tensors."""

import os
import random
import urllib.request
import zipfile

import torch
from PIL import Image
from torchvision import transforms

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMG_SIZE = 224

# Shorter side to 224, then center crop: every image becomes a 224x224 tensor with values in [0, 1].
_to_tensor = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


def download_coco(data_dir):
    """Download and unzip COCO val2017 (5000 images, ~800 MB) into data_dir, once."""
    img_dir = os.path.join(data_dir, "val2017")
    if os.path.isdir(img_dir) and len(os.listdir(img_dir)) == 5000:
        return img_dir
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "val2017.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {COCO_URL} ...")
        urllib.request.urlretrieve(COCO_URL, zip_path)
    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return img_dir


def make_splits(img_dir, n_whiten, n_mark, n_negative, seed=0):
    """Shuffle the image list with a fixed seed and cut it into 3 disjoint splits.

    - whiten:   unmarked images used to fit the PCA-whitening
    - mark:     images we watermark and then attack
    - negative: unmarked images used to check that we get no false positives
    """
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    assert n_whiten + n_mark + n_negative <= len(files), "not enough images for disjoint splits"
    random.Random(seed).shuffle(files)
    paths = [os.path.join(img_dir, f) for f in files]
    return {
        "whiten": paths[:n_whiten],
        "mark": paths[n_whiten:n_whiten + n_mark],
        
        "negative": paths[n_whiten + n_mark:n_whiten + n_mark + n_negative],
    }


def load_image(path):
    """Load one image as a (3, 224, 224) float tensor in [0, 1] (grayscale images become RGB)."""
    return _to_tensor(Image.open(path).convert("RGB"))


def iterate_batches(paths, batch_size=64):
    """Yield batches of images (B, 3, 224, 224) so we never hold thousands of images in memory."""
    for i in range(0, len(paths), batch_size):
        yield torch.stack([load_image(p) for p in paths[i:i + batch_size]])
