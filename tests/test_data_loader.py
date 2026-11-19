"""Tests for image loading (uses temporary synthetic images, no real data needed)."""

import numpy as np
import pytest
from PIL import Image

from utils import load_dataset, load_image, load_sample_images


def _write_png(path, arr):
    Image.fromarray(arr).save(path)


def test_load_image_gray_scaled_to_unit_range(tmp_path):
    arr = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    p = tmp_path / "a.png"
    _write_png(p, arr)
    out = load_image(p)
    assert out.dtype == np.float64 and out.shape == (2, 2)
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.allclose(out, arr / 255.0)


def test_load_image_converts_colour_to_gray(tmp_path):
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red
    p = tmp_path / "red.png"
    _write_png(p, rgb)
    out = load_image(p)
    assert out.shape == (4, 6)  # 2D
    assert np.allclose(out, 0.299, atol=0.01)  # luma of pure red


def test_load_dataset_sorted_and_format(tmp_path):
    folder = tmp_path / "set12"
    folder.mkdir()
    for name in ["b.png", "a.png", "c.png"]:
        _write_png(folder / name, np.full((8, 8), 100, dtype=np.uint8))
    (folder / "notes.txt").write_text("ignore me")

    data = load_dataset("set12", data_root=tmp_path)
    assert [n for n, _ in data] == ["a.png", "b.png", "c.png"]
    for _, img in data:
        assert img.dtype == np.float64 and img.ndim == 2
        assert 0.0 <= img.min() and img.max() <= 1.0

    assert len(load_dataset("Set12", data_root=tmp_path, max_images=2)) == 2


def test_load_dataset_errors(tmp_path):
    with pytest.raises(ValueError):
        load_dataset("not-a-dataset", data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset("kodak24", data_root=tmp_path)


def test_sample_images_follow_the_contract():
    imgs = load_sample_images(size=128)
    assert len(imgs) >= 3
    for name, img in imgs:
        assert isinstance(name, str)
        assert img.dtype == np.float64 and img.ndim == 2
        assert 0.0 <= img.min() and img.max() <= 1.0
