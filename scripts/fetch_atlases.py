#!/usr/bin/env python3
"""Download HCP-MMP1 fsaverage5 annot files into ``atlases/``.

Idempotent: skips files that already exist with non-zero size.

Source: HCP-MMP1 0 projected on fsaverage (Mills 2016 Figshare mirror).
fsaverage5 versions are produced by resampling fsaverage7 with
``mri_surf2surf``; see ``docs/INTERPRETATION.md`` for the upstream pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("fetch_atlases")

# Placeholder URLs. The Figshare mirror at
# https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446
# distributes fsaverage7 annot files; fsaverage5 versions are derived locally
# or via the FreeSurfer mri_surf2surf pipeline. Replace these with the
# canonical URLs once the project repo is published.
_LH_URL = "https://figshare.com/ndownloader/files/5528816"  # TODO: fsaverage5 LH annot
_RH_URL = "https://figshare.com/ndownloader/files/5528819"  # TODO: fsaverage5 RH annot
_MIN_BYTES = 50_000  # annot files are ~80 KB; reject obvious failures


def _download(url: str, dest: Path) -> None:
    logger.info("downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _validate(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < _MIN_BYTES:
        raise RuntimeError(f"{path} suspiciously small ({size} bytes)")
    # FreeSurfer annot files start with a 4-byte magic
    with open(path, "rb") as f:
        magic = f.read(4)
    if not magic:
        raise RuntimeError(f"{path} could not be read")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "atlases",
        help="directory to write annot files into",
    )
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    targets = {
        "lh.HCP-MMP1.annot": _LH_URL,
        "rh.HCP-MMP1.annot": _RH_URL,
    }
    for name, url in targets.items():
        dest = args.atlas_dir / name
        if dest.exists() and dest.stat().st_size >= _MIN_BYTES and not args.force:
            logger.info("OK %s already present (%d bytes); skipping", dest, dest.stat().st_size)
            continue
        try:
            _download(url, dest)
            _validate(dest)
            logger.info("OK %s validated (%d bytes)", dest, dest.stat().st_size)
        except Exception as exc:
            logger.error("FAIL failed to fetch %s: %s", name, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
