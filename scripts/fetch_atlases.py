#!/usr/bin/env python3
"""Download HCP-MMP1 fsaverage5 annot files into ``atlases/``.

Idempotent: skips files that already exist with non-zero size.

Source: the ``poldrack/GOBS`` repository ships native-fsaverage5 GIFTI label
files (10242 vertices per hemisphere). We download those and convert to
FreeSurfer ``.annot`` so ``nibabel.freesurfer.read_annot`` can consume them.

The widely-cited Figshare mirror at 3498446 distributes fsaverage7 (163842
vertices) and is NOT a drop-in for our fsaverage5 pipeline — see
docs/INTERPRETATION.md for context.

Citations:
- Glasser et al. 2016, Nature 536:171–178 (HCP-MMP1 parcellation)
- GOBS repository (poldrack/GOBS) for the fsaverage5 projection
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger("fetch_atlases")

_GOBS_BASE = "https://raw.githubusercontent.com/poldrack/GOBS/master/extract/HCP-MMP1"
_LH_GII_URL = f"{_GOBS_BASE}/lh.HCP-MMP1.fsaverage5.gii"
_RH_GII_URL = f"{_GOBS_BASE}/rh.HCP-MMP1.fsaverage5.gii"
_MIN_BYTES = 10_000  # the GIFTI files are ~29 KB; .annot output is similar
_VERTS_PER_HEMI = 10242


def _download(url: str, dest: Path) -> None:
    logger.info("downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _gii_to_annot(gii_path: Path, annot_path: Path) -> None:
    """Convert a HCP-MMP1 fsaverage5 GIFTI label file to FreeSurfer annot.

    Raises:
        RuntimeError: if the vertex count is not 10242 (fsaverage5).
    """
    gii = nib.load(str(gii_path))
    labels = gii.darrays[0].data.astype(np.int32)
    if labels.shape != (_VERTS_PER_HEMI,):
        raise RuntimeError(
            f"{gii_path} has shape {labels.shape}; expected ({_VERTS_PER_HEMI},) " "for fsaverage5"
        )

    n_labels = len(gii.labeltable.labels)
    ctab = np.zeros((n_labels, 5), dtype=np.int32)
    names: list[bytes] = []
    for i, lbl in enumerate(gii.labeltable.labels):
        r, g, b, _ = lbl.rgba
        ctab[i, 0] = int(round(r * 255))
        ctab[i, 1] = int(round(g * 255))
        ctab[i, 2] = int(round(b * 255))
        ctab[i, 4] = ctab[i, 0] + (ctab[i, 1] << 8) + (ctab[i, 2] << 16)
        names.append(lbl.label.encode())

    nib.freesurfer.write_annot(str(annot_path), labels, ctab, names)
    logger.info("OK %s written (%d labels, %d vertices)", annot_path, n_labels, labels.shape[0])


def _validate(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < _MIN_BYTES:
        raise RuntimeError(f"{path} suspiciously small ({size} bytes)")
    labels, _, _ = nib.freesurfer.read_annot(str(path))
    if labels.shape != (_VERTS_PER_HEMI,):
        raise RuntimeError(f"{path} has shape {labels.shape}; expected ({_VERTS_PER_HEMI},)")


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
        "lh.HCP-MMP1.annot": _LH_GII_URL,
        "rh.HCP-MMP1.annot": _RH_GII_URL,
    }
    for name, url in targets.items():
        dest = args.atlas_dir / name
        if dest.exists() and dest.stat().st_size >= _MIN_BYTES and not args.force:
            logger.info("OK %s already present (%d bytes); skipping", dest, dest.stat().st_size)
            continue
        gii_tmp = args.atlas_dir / Path(url).name
        try:
            _download(url, gii_tmp)
            _gii_to_annot(gii_tmp, dest)
            _validate(dest)
            logger.info("OK %s validated (%d bytes)", dest, dest.stat().st_size)
        except Exception as exc:
            logger.error("FAIL failed to fetch %s: %s", name, exc)
            return 1
        finally:
            if gii_tmp.exists():
                gii_tmp.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
