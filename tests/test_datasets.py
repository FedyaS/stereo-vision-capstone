"""Dataset loaders, including the one property that matters most: the
synthetic pair's ground truth must be exact, not approximately right."""
from __future__ import annotations

import numpy as np
import pytest

from bench.datasets import (
    list_middlebury_scenes,
    load_image_pair,
    make_synthetic_pair,
    read_pfm,
    resolve_pairs,
)
from bench.datasets.middlebury import parse_calib, write_pfm
from bench.paths import PRACTICE_DIR


def test_synthetic_pair_is_self_consistent():
    """Left(y, x) must equal Right(y, x - d) at the ground-truth disparity.

    This verifies the generator's convention directly. If it were inverted,
    every algorithm would score terribly on synthetic data and the test suite
    would be blaming the wrong component.
    """
    import cv2

    pair = make_synthetic_pair(seed=2, size=(120, 200), ndisp=32)
    h, w = pair.shape
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    resampled = cv2.remap(
        pair.right, xx - pair.gt, yy,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
    )
    interior = pair.occ_mask & (xx > pair.ndisp + 2)
    assert interior.sum() > 1000
    diff = np.abs(
        resampled[interior].astype(np.float32) - pair.left[interior].astype(np.float32)
    )
    assert diff.mean() < 1e-3


def test_synthetic_pair_shapes_and_ranges():
    pair = make_synthetic_pair(seed=1, size=(64, 96), ndisp=16)
    assert pair.left.shape == (64, 96, 3)
    assert pair.right.shape == (64, 96, 3)
    assert pair.left.dtype == np.uint8
    assert pair.has_gt
    assert pair.gt.dtype == np.float32
    assert pair.gt.min() >= 0
    assert pair.gt.max() < pair.ndisp
    assert pair.megapixels == pytest.approx(64 * 96 / 1e6)


def test_synthetic_is_reproducible_and_seed_dependent():
    a = make_synthetic_pair(seed=4, size=(48, 64))
    b = make_synthetic_pair(seed=4, size=(48, 64))
    c = make_synthetic_pair(seed=5, size=(48, 64))
    assert np.array_equal(a.left, b.left)
    assert np.array_equal(a.gt, b.gt)
    assert not np.array_equal(a.gt, c.gt)


def test_synthetic_eval_mask_excludes_left_border():
    pair = make_synthetic_pair(seed=0, size=(64, 128), ndisp=24)
    # No right-image match can exist within dmax of the left edge.
    assert not pair.occ_mask[:, :20].any()


def test_pfm_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    data = (rng.random((17, 23)) * 50).astype(np.float32)
    path = tmp_path / "d.pfm"
    write_pfm(path, data)
    assert np.allclose(read_pfm(path), data)


def test_pfm_rejects_non_pfm(tmp_path):
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"NOTPFM\n1 1\n-1.0\n")
    with pytest.raises(ValueError):
        read_pfm(path)


def test_pfm_preserves_row_order(tmp_path):
    # PFM stores bottom-to-top; a round trip must not flip the image.
    data = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    path = tmp_path / "o.pfm"
    write_pfm(path, data)
    assert np.array_equal(read_pfm(path), data)


def test_parse_calib(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text(
        "cam0=[1500 0 199; 0 1500 187; 0 0 1]\n"
        "doffs=52\nbaseline=80\nndisp=64\n"
    )
    calib = parse_calib(p)
    assert calib["ndisp"] == 64.0
    assert calib["baseline"] == 80.0
    assert isinstance(calib["cam0"], str)


def test_resolve_pairs_synthetic_count():
    assert len(resolve_pairs("synthetic:3")) == 3
    assert len(resolve_pairs("synthetic")) == 1
    assert len(resolve_pairs("synthetic:5", limit=2)) == 2


def test_local_loader_reads_practice_folder():
    """The committed sample pair, which doubles as the folder-dataset example.

    Asserted rather than skipped: these two PNGs are in the repository, so an
    absent folder means something moved and the docs and ``bench.paths`` now
    point at nothing. A skip here let exactly that pass unnoticed.
    """
    folder = PRACTICE_DIR
    assert (folder / "im0.png").exists(), (
        f"{folder} has no im0.png -- the committed sample pair moved; "
        "update bench.paths.PRACTICE_DIR and the docs that cite it."
    )
    pair = load_image_pair(folder, downscale=8)
    assert pair.left.shape == pair.right.shape
    assert pair.left.shape[0] < 400
    assert pair.ndisp >= 16


def test_local_loader_rejects_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_image_pair(tmp_path / "nope")


def test_local_loader_rejects_folder_without_images(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_pair(tmp_path)


# --------------------------------------------------------------------------
# Middlebury, only when the download has been run
# --------------------------------------------------------------------------


def test_middlebury_scene_loads(teddy):
    assert teddy.name == "middlebury/Teddy"
    assert teddy.has_gt
    assert teddy.occ_mask is not None
    assert teddy.ndisp == 64
    assert teddy.left.shape == teddy.right.shape
    assert teddy.gt.shape == teddy.left.shape[:2]
    # Ground truth is sparse-by-nan, never inf.
    assert not np.isinf(teddy.gt).any()
    known = np.isfinite(teddy.gt)
    assert known.mean() > 0.9
    assert teddy.gt[known].min() >= 0
    assert teddy.focal_length_px == pytest.approx(1500.0)


def test_middlebury_ndisp_covers_ground_truth(middlebury_scenes):
    """The scene's stated search range must actually contain its disparities."""
    from bench.datasets import load_middlebury_scene

    if not middlebury_scenes:
        pytest.skip("Middlebury not downloaded")
    for name in middlebury_scenes[:6]:
        pair = load_middlebury_scene(name)
        known = np.isfinite(pair.gt)
        assert pair.gt[known].max() <= pair.ndisp, name


def test_middlebury_missing_scene_message():
    from bench.datasets import load_middlebury_scene

    with pytest.raises(FileNotFoundError) as exc:
        load_middlebury_scene("NotAScene")
    assert "get_dataset" in str(exc.value)


# --------------------------------------------------------------------------
# folders of pairs, at any depth
# --------------------------------------------------------------------------


def test_find_pair_dirs_handles_every_layout(tmp_path):
    """One notion of "a folder of images", whatever shape it has on disk."""
    import cv2

    from bench.datasets import find_pair_dirs, is_pair_dir

    def pair(folder, left="im0", right="im1", ext=".png"):
        folder.mkdir(parents=True, exist_ok=True)
        img = np.zeros((8, 8, 3), np.uint8)
        cv2.imwrite(str(folder / f"{left}{ext}"), img)
        cv2.imwrite(str(folder / f"{right}{ext}"), img)

    # a single pair: the folder itself, not its children
    single = tmp_path / "single"
    pair(single)
    assert is_pair_dir(single)
    assert find_pair_dirs(single) == [single]

    # flat
    flat = tmp_path / "flat"
    for i in range(3):
        pair(flat / f"{i:05d}")
    assert len(find_pair_dirs(flat)) == 3

    # splits, and one level deeper than that
    deep = tmp_path / "deep"
    pair(deep / "run1" / "train" / "00000")
    pair(deep / "run1" / "val" / "00000", left="left", right="right", ext=".jpg")
    found = find_pair_dirs(deep)
    assert [p.name for p in found] == ["00000", "00000"]
    assert {p.parent.name for p in found} == {"train", "val"}

    # the order is stable and a limit takes a prefix of it
    assert find_pair_dirs(flat, limit=2) == find_pair_dirs(flat)[:2]

    # a folder with no pair in it is empty, not an error
    (tmp_path / "empty" / "nothing").mkdir(parents=True)
    assert find_pair_dirs(tmp_path / "empty") == []


def test_load_pair_folder_names_pairs_by_their_path(tmp_path):
    """A result row has to say which sample it came from."""
    import cv2

    from bench.datasets import load_pair_folder

    for split in ("test", "train"):
        d = tmp_path / "set" / split / "00000"
        d.mkdir(parents=True)
        img = np.zeros((8, 8, 3), np.uint8)
        cv2.imwrite(str(d / "im0.png"), img)
        cv2.imwrite(str(d / "im1.png"), img)

    pairs = load_pair_folder(tmp_path / "set")
    assert [p.name for p in pairs] == ["set/test/00000", "set/train/00000"]


def test_resolve_pairs_accepts_a_nested_folder(tmp_path):
    """``-d <path>`` works on a dataset copied in by hand, not just on the
    layouts the generator happens to write."""
    import cv2

    from bench.datasets import resolve_pairs

    for i in range(2):
        d = tmp_path / "set" / "run1" / "train" / f"{i:05d}"
        d.mkdir(parents=True)
        img = np.zeros((8, 8, 3), np.uint8)
        cv2.imwrite(str(d / "im0.png"), img)
        cv2.imwrite(str(d / "im1.png"), img)

    assert len(resolve_pairs(str(tmp_path / "set"))) == 2
    assert len(resolve_pairs(str(tmp_path / "set"), limit=1)) == 1
