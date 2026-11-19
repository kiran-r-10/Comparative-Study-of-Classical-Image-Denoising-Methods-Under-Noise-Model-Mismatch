"""Image loading: files on disk -> float64 grayscale arrays in [0, 1].

Contract: ``load_dataset(name)`` returns ``list[(filename, ndarray)]`` where each
array has shape (H, W), dtype float64 and values in [0, 1].
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import PROJECT_ROOT

DATASET_NAMES = ("set12", "kodak24", "bsd68")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_image(path):
    """Load one image as grayscale float64 in [0, 1].

    Colour images are converted with PIL's ITU-R 601 luma
    (L = 0.299 R + 0.587 G + 0.114 B). 8-bit input is assumed.
    """
    with Image.open(path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float64)
    return arr / 255.0


def load_dataset(name, data_root=None, max_images=None):
    """Load all images of a dataset folder ``<data_root>/<name>/``.

    Parameters
    ----------
    name : one of "set12", "kodak24", "bsd68"
    data_root : folder containing the dataset folders (default: ``<repo>/data``)
    max_images : optionally only load the first N images (sorted by filename)
    """
    key = name.lower()
    if key not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}.")

    root = Path(data_root) if data_root is not None else PROJECT_ROOT / "data"
    folder = root / key
    files = []
    if folder.is_dir():
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No images found in '{folder}'. Download the dataset as described "
            f"in data/README.md and put the image files in that folder."
        )
    if max_images is not None:
        files = files[:max_images]
    return [(p.name, load_image(p)) for p in files]


def load_sample_images(size=256):
    """Fallback test images bundled with scikit-image (no download needed).

    Returns ``list[(name, ndarray)]`` in the same format as ``load_dataset``,
    center-cropped to ``size x size``. Use only until the real datasets are in
    place.
    """
    from skimage import data

    names = ["camera", "coins", "moon", "page", "astronaut", "chelsea"]
    out = []
    for n in names:
        arr = getattr(data, n)()
        gray = Image.fromarray(arr).convert("L")
        a = np.asarray(gray, dtype=np.float64) / 255.0
        h, w = a.shape
        s = min(size, h, w)
        t, l = (h - s) // 2, (w - s) // 2
        out.append((n, a[t : t + s, l : l + s]))
    return out
