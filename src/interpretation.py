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


def _normalise_parcel(name: str) -> str:
    """Strip HCP-MMP1 ``_ROI`` suffix and leading hemi prefix.

    The GOBS annot files ship labels like ``L_V1_ROI``; ``parcel_lookup.json``
    uses bare names like ``V1``. We normalise to the bare form so the loop can
    re-prefix with the correct hemisphere.
    """
    s = name
    if s.endswith("_ROI"):
        s = s[: -len("_ROI")]
    if s.startswith(("L_", "R_")):
        s = s[2:]
    return s


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

        Each row has: ``parcel`` (HCP-MMP1 code), ``name`` (plain-English),
        ``category`` (functional grouping), ``function`` (one-sentence
        layman description), ``activation`` (float, signed), ``terms`` (legacy
        Neurosynth terms). Sorted by ``|activation|`` descending. If annot
        files are missing the return is empty.
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
                key = f"{hemi}_{_normalise_parcel(name)}"
                activation = float(pred[mask].mean())
                entry = self._lookup.get(key, {})
                rows.append(
                    {
                        "parcel": key,
                        "name": entry.get("name", key),
                        "category": entry.get("category"),
                        "function": entry.get("function", ""),
                        "activation": activation,
                        "terms": list(entry.get("terms", [])),
                    }
                )

        rows.sort(key=lambda r: abs(r["activation"]), reverse=True)
        return rows[:k]

    # -- summarisation ----------------------------------------------------

    # Coarse functional categories surfaced to layman readers. The display
    # name and a one-sentence prose template land in the summary. Keep this
    # list in sync with the categories declared in atlases/parcel_lookup.json.
    _CATEGORY_DISPLAY: dict[str, str] = {
        "vision_low": "early visual cortex",
        "vision_ventral": "ventral visual stream (objects, faces, scenes)",
        "motion": "visual motion areas",
        "auditory": "auditory cortex",
        "language": "language network",
        "somatomotor": "somatosensory and motor cortex",
        "attention": "attention network",
        "salience": "salience/insula network",
        "default_mode": "default-mode network",
        "executive": "cognitive-control network",
        "reward_emotion": "reward and emotion areas",
        "social_cognition": "social-cognition network",
    }

    _CATEGORY_BLURB: dict[str, str] = {
        "vision_low": "the model expects you to be seeing simple visual features — edges, contrast, orientation.",
        "vision_ventral": "the model expects you to be recognising objects, faces, or scenes.",
        "motion": "the model expects you to be perceiving movement.",
        "auditory": "the model expects you to be listening to sound (speech, music, or environmental audio).",
        "language": "the model expects linguistic processing — words, syntax, meaning.",
        "somatomotor": "the model expects body sensation or movement.",
        "attention": "the model expects effortful attention or spatial orienting.",
        "salience": "the model expects salience or interoceptive signalling (body-state awareness).",
        "default_mode": "the model expects internal/self-referential thought — mind wandering, memory recall, mental imagery.",
        "executive": "the model expects cognitive control — conflict, planning, working memory.",
        "reward_emotion": "the model expects reward, value, or emotional processing.",
        "social_cognition": "the model expects social inference — thinking about other minds, biological motion.",
    }

    def summarize(self, top_regions: list[dict[str, Any]]) -> dict[str, Any]:
        """Translate a top-K region list into a layman-friendly summary.

        Returns ``{"headline": str, "narrative": str, "categories": list[dict]}``:

        - ``headline``: one-line description of the dominant network.
        - ``narrative``: 2–4 sentences expanding the headline + secondary
          activations + caveat about population-averaged BOLD.
        - ``categories``: ranked breakdown ``[{"category", "display", "count",
          "mean_activation", "max_activation"}]`` so the UI can render a
          bar/badge view if desired.

        Falls back to a neutral placeholder if ``top_regions`` is empty (e.g.
        atlas in stub mode).
        """
        if not top_regions:
            return {
                "headline": "No region table available.",
                "narrative": (
                    "The HCP-MMP1 atlas isn't loaded, so the prediction can't "
                    "be summarised. Run `scripts/fetch_atlases.py`."
                ),
                "categories": [],
            }

        # Group rows by category. Drop rows whose category is missing from the
        # lookup so we don't claim more knowledge than we have.
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in top_regions:
            cat = row.get("category")
            if not cat:
                continue
            buckets.setdefault(cat, []).append(row)

        cats: list[dict[str, Any]] = []
        for cat, rows in buckets.items():
            acts = [r["activation"] for r in rows]
            cats.append(
                {
                    "category": cat,
                    "display": self._CATEGORY_DISPLAY.get(cat, cat),
                    "count": len(rows),
                    "mean_activation": float(np.mean(acts)),
                    "max_activation": float(max(acts, key=abs)),
                }
            )

        # Rank by combined (count, absolute mean activation) so a 4-region
        # auditory pattern with mean 0.30 beats a 1-region motor pattern with
        # mean 0.45.
        cats.sort(
            key=lambda c: (c["count"], abs(c["mean_activation"])),
            reverse=True,
        )

        if not cats:
            return {
                "headline": "Pattern doesn't fall into a known network.",
                "narrative": (
                    "Top regions are present but none of them are in the "
                    "v0 curated category map. See the table below for the "
                    "raw HCP-MMP1 codes."
                ),
                "categories": [],
            }

        primary = cats[0]
        primary_label = primary["display"]
        primary_blurb = self._CATEGORY_BLURB.get(primary["category"], "")
        sign_word = "positive" if primary["mean_activation"] > 0 else "negative"
        headline = (
            f"Dominant network: **{primary_label}** "
            f"({primary['count']} of the top regions; "
            f"mean {sign_word} activation {primary['mean_activation']:+.2f})."
        )

        narrative_parts = [headline.replace("**", "")]
        if primary_blurb:
            narrative_parts.append("That is, " + primary_blurb)

        if len(cats) >= 2:
            secondary = cats[1]
            narrative_parts.append(
                f"Secondary activation in {secondary['display']} "
                f"({secondary['count']} region"
                f"{'s' if secondary['count'] != 1 else ''})."
            )

        narrative_parts.append(
            "Predictions are population-averaged BOLD at ~1 Hz with a ~5 s "
            "hemodynamic lag; treat as a hypothesis about typical responses, "
            "not a measurement of any specific person."
        )

        return {
            "headline": headline,
            "narrative": " ".join(narrative_parts),
            "categories": cats,
        }
