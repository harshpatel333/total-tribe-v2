"""HCP-MMP1 region interpretation for TRIBE v2 predictions.

Loads fsaverage5-resampled HCP-MMP1 annotations and a parcel→Neurosynth-terms
lookup, then aggregates a per-vertex activation vector into a top-K region
table. Pure CPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# fsaverage5 has 10242 vertices per hemisphere → 20484 total
_VERTS_PER_HEMI = 10242


class RegionInterpreter:
    """Map a per-vertex activation vector into a top-K HCP-MMP1 region table.

    Parameters
    ----------
    atlas_dir:
        Directory containing ``lh.HCP-MMP1.annot``, ``rh.HCP-MMP1.annot``,
        and ``parcel_lookup.json``. See ``scripts/fetch_atlases.py`` and
        ``scripts/build_neurosynth_lookup.py``.
    """

    def __init__(self, atlas_dir: Path) -> None:
        self.atlas_dir = Path(atlas_dir)
        self._lh_labels: np.ndarray | None = None
        self._rh_labels: np.ndarray | None = None
        self._lh_names: list[str] = []
        self._rh_names: list[str] = []
        self._lookup: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        lh_annot = self.atlas_dir / "lh.HCP-MMP1.annot"
        rh_annot = self.atlas_dir / "rh.HCP-MMP1.annot"
        lookup_path = self.atlas_dir / "parcel_lookup.json"

        # TODO: in production, both annot files must exist. For scaffolding,
        # missing files are degraded to a stub state with a clear warning so
        # the UI and tests still load.
        try:
            import nibabel.freesurfer  # imported lazily — heavy

            lh_labels, _, lh_names = nibabel.freesurfer.read_annot(str(lh_annot))
            rh_labels, _, rh_names = nibabel.freesurfer.read_annot(str(rh_annot))
            self._lh_labels = lh_labels
            self._rh_labels = rh_labels
            self._lh_names = [n.decode() if isinstance(n, bytes) else n for n in lh_names]
            self._rh_names = [n.decode() if isinstance(n, bytes) else n for n in rh_names]
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "HCP-MMP1 annot files missing under %s (%s); RegionInterpreter "
                "is running in stub mode. Run scripts/fetch_atlases.py.",
                self.atlas_dir,
                exc,
            )

        try:
            with open(lookup_path, encoding="utf-8") as f:
                self._lookup = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "parcel_lookup.json missing at %s; terms will be empty.",
                lookup_path,
            )
            self._lookup = {}

    def top_regions(
        self,
        pred_t: np.ndarray,
        k: int = 8,
    ) -> list[dict[str, Any]]:
        """Aggregate ``pred_t`` (shape ``(20484,)``) into a top-K parcel list.

        Returns a list of ``{"parcel": str, "activation": float, "terms": list[str]}``
        sorted by ``|activation|`` descending. If annot files are missing the
        return is empty.
        """
        if pred_t.ndim != 1 or pred_t.shape[0] != 2 * _VERTS_PER_HEMI:
            raise ValueError(f"pred_t must be shape ({2 * _VERTS_PER_HEMI},), got {pred_t.shape}")

        if self._lh_labels is None or self._rh_labels is None:
            logger.warning("top_regions called in stub mode; returning empty list.")
            return []

        lh_pred = pred_t[:_VERTS_PER_HEMI]
        rh_pred = pred_t[_VERTS_PER_HEMI:]

        rows: list[dict[str, Any]] = []
        for hemi, labels, names, pred in (
            ("L", self._lh_labels, self._lh_names, lh_pred),
            ("R", self._rh_labels, self._rh_names, rh_pred),
        ):
            for idx, name in enumerate(names):
                if not name or name in {"???", "Unknown", "Medial_Wall"}:
                    continue
                mask = labels == idx
                if not mask.any():
                    continue
                key = f"{hemi}_{name}"
                activation = float(pred[mask].mean())
                entry = self._lookup.get(key, {})
                rows.append(
                    {
                        "parcel": key,
                        "activation": activation,
                        "terms": list(entry.get("terms", [])),
                    }
                )

        rows.sort(key=lambda r: abs(r["activation"]), reverse=True)
        return rows[:k]
