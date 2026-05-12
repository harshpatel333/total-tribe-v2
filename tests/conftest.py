"""Shared pytest fixtures for total-tribe-v2 tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_VERTS_PER_HEMI = 10242


@pytest.fixture
def fake_pred() -> np.ndarray:
    """A deterministic synthetic activation vector of length 20484."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(2 * _VERTS_PER_HEMI).astype(np.float32)


@pytest.fixture
def fake_lookup() -> dict:
    """Minimal parcel_lookup-shaped dict for interpretation tests."""
    return {
        "L_FFC": {"terms": ["faces"], "score": 0.9},
        "R_FFC": {"terms": ["faces"], "score": 0.9},
        "L_V1": {"terms": ["vision"], "score": 0.8},
    }


@pytest.fixture
def fake_atlas_dir(tmp_path: Path, fake_lookup: dict) -> Path:
    """Returns an atlas_dir with ONLY parcel_lookup.json present.

    Annot files are intentionally absent so we exercise the stub-mode path.
    """
    d = tmp_path / "atlases"
    d.mkdir()
    (d / "parcel_lookup.json").write_text(json.dumps(fake_lookup), encoding="utf-8")
    return d


@pytest.fixture
def fake_atlas_dir_with_annot(
    tmp_path: Path,
    fake_lookup: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Returns an atlas_dir where ``nibabel.freesurfer.read_annot`` is monkeypatched.

    The synthetic annot data has:
      LH parcels: "???", "FFC", "V1"   (indices 0, 1, 2)
      RH parcels: "???", "FFC"          (indices 0, 1)

    Half of LH vertices belong to FFC, the rest to V1.
    All RH vertices belong to FFC.

    This lets the interpretation tests assert ordering and term lookup without
    real ``.annot`` files on disk.
    """
    d = tmp_path / "atlases"
    d.mkdir()
    (d / "parcel_lookup.json").write_text(json.dumps(fake_lookup), encoding="utf-8")
    # touch the annot files so RegionInterpreter._load() doesn't fall through
    (d / "lh.HCP-MMP1.annot").write_bytes(b"\x00")
    (d / "rh.HCP-MMP1.annot").write_bytes(b"\x00")

    lh_labels = np.zeros(_VERTS_PER_HEMI, dtype=np.int32)
    lh_labels[: _VERTS_PER_HEMI // 2] = 1  # FFC
    lh_labels[_VERTS_PER_HEMI // 2 :] = 2  # V1
    lh_names = ["???", "FFC", "V1"]

    rh_labels = np.ones(_VERTS_PER_HEMI, dtype=np.int32)  # all FFC
    rh_names = ["???", "FFC"]

    def fake_read_annot(path):  # noqa: ANN001
        if "lh" in Path(path).name:
            return lh_labels, np.zeros((3, 5), dtype=np.int32), lh_names
        return rh_labels, np.zeros((2, 5), dtype=np.int32), rh_names

    # Patch the import path used in src.interpretation
    import nibabel.freesurfer

    monkeypatch.setattr(nibabel.freesurfer, "read_annot", fake_read_annot)
    return d
