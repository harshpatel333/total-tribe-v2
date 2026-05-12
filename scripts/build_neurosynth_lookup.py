#!/usr/bin/env python3
"""Generate ``atlases/parcel_lookup.json`` from Neurosynth via NiMARE.

For each HCP-MMP1 parcel centroid, queries the Neurosynth term-association
dataset for the top-K cognitive terms above a minimum score threshold.

Build-time only. The hand-curated ``atlases/parcel_lookup.json`` that ships in
the repo is sufficient for the v1 demo; running this script regenerates it
from data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_neurosynth_lookup")

_DEFAULT_K = 5
_DEFAULT_MIN_SCORE = 0.05


def _hand_curated_fallback() -> dict[str, dict]:
    """Used if NiMARE is unavailable. Mirror of ``atlases/parcel_lookup.json``."""
    fallback_path = Path(__file__).resolve().parent.parent / "atlases" / "parcel_lookup.json"
    if fallback_path.exists():
        with open(fallback_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build(
    out_path: Path,
    k: int = _DEFAULT_K,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> int:
    try:
        import nimare  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "NiMARE not available; writing hand-curated fallback to %s",
            out_path,
        )
        lookup = _hand_curated_fallback()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2, sort_keys=True)
        return 0

    # TODO: real NiMARE pipeline. Sketch:
    #   1. nimare.extract.fetch_neurosynth(...)
    #   2. nimare.io.convert_neurosynth_to_dataset(...)
    #   3. For each HCP-MMP1 parcel: extract parcel centroid (MNI), call
    #      nimare.decode.continuous.CorrelationDecoder against the parcel mask.
    #   4. Keep top-K terms with score >= min_score; serialize.
    _ = (k, min_score)  # consumed by the future NiMARE pipeline above
    logger.error(
        "Full NiMARE pipeline is not yet implemented; this script ships as a "
        "scaffold. See docs/INTERPRETATION.md and TODO inline."
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "atlases" / "parcel_lookup.json",
    )
    parser.add_argument("--k", type=int, default=_DEFAULT_K)
    parser.add_argument("--min-score", type=float, default=_DEFAULT_MIN_SCORE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    return build(args.out, args.k, args.min_score)


if __name__ == "__main__":
    sys.exit(main())
