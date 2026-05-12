"""Unit tests for ``src.interpretation.RegionInterpreter``.

Pure CPU. Uses monkeypatched ``nibabel.freesurfer.read_annot`` fixtures from
``conftest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import interpretation

_VERTS_PER_HEMI = 10242


def test_top_regions_missing_annot_returns_empty(
    fake_atlas_dir: Path,
    fake_pred: np.ndarray,
) -> None:
    interp = interpretation.RegionInterpreter(atlas_dir=fake_atlas_dir)
    rows = interp.top_regions(fake_pred, k=8)
    assert rows == []


def test_top_regions_with_synthetic_annot(
    fake_atlas_dir_with_annot: Path,
) -> None:
    interp = interpretation.RegionInterpreter(atlas_dir=fake_atlas_dir_with_annot)

    pred = np.zeros(2 * _VERTS_PER_HEMI, dtype=np.float32)
    # LH FFC half: strong positive
    pred[: _VERTS_PER_HEMI // 2] = 2.0
    # LH V1 half: weak positive
    pred[_VERTS_PER_HEMI // 2 : _VERTS_PER_HEMI] = 0.1
    # RH FFC all: medium negative
    pred[_VERTS_PER_HEMI:] = -1.0

    rows = interp.top_regions(pred, k=8)
    assert len(rows) == 3
    parcels = [r["parcel"] for r in rows]
    # Order by |activation| desc -> L_FFC (2.0), R_FFC (1.0), L_V1 (0.1)
    assert parcels == ["L_FFC", "R_FFC", "L_V1"]
    assert rows[0]["terms"] == ["faces"]
    assert rows[1]["terms"] == ["faces"]
    assert rows[2]["terms"] == ["vision"]
    assert rows[0]["activation"] == pytest.approx(2.0)
    assert rows[1]["activation"] == pytest.approx(-1.0)
    assert rows[2]["activation"] == pytest.approx(0.1)


def test_top_regions_k_truncates(
    fake_atlas_dir_with_annot: Path,
) -> None:
    interp = interpretation.RegionInterpreter(atlas_dir=fake_atlas_dir_with_annot)
    pred = np.linspace(-1, 1, 2 * _VERTS_PER_HEMI).astype(np.float32)
    rows = interp.top_regions(pred, k=2)
    assert len(rows) == 2


def test_top_regions_rejects_wrong_shape(fake_atlas_dir: Path) -> None:
    interp = interpretation.RegionInterpreter(atlas_dir=fake_atlas_dir)
    with pytest.raises(ValueError, match="must be shape"):
        interp.top_regions(np.zeros(10), k=4)


def test_missing_parcel_in_lookup_returns_empty_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the lookup is empty, terms list per row should be empty."""
    d = tmp_path / "atlases"
    d.mkdir()
    (d / "parcel_lookup.json").write_text(json.dumps({}), encoding="utf-8")
    (d / "lh.HCP-MMP1.annot").write_bytes(b"\x00")
    (d / "rh.HCP-MMP1.annot").write_bytes(b"\x00")

    lh_labels = np.ones(_VERTS_PER_HEMI, dtype=np.int32)  # all FFC
    rh_labels = np.zeros(_VERTS_PER_HEMI, dtype=np.int32)  # all unknown
    names = ["???", "FFC"]

    def fake_read_annot(path):  # noqa: ANN001
        if "lh" in Path(path).name:
            return lh_labels, np.zeros((2, 5), dtype=np.int32), names
        return rh_labels, np.zeros((2, 5), dtype=np.int32), names

    import nibabel.freesurfer

    monkeypatch.setattr(nibabel.freesurfer, "read_annot", fake_read_annot)

    interp = interpretation.RegionInterpreter(atlas_dir=d)
    pred = np.ones(2 * _VERTS_PER_HEMI, dtype=np.float32)
    rows = interp.top_regions(pred, k=4)
    assert len(rows) == 1
    assert rows[0]["parcel"] == "L_FFC"
    assert rows[0]["terms"] == []
