"""Inference engine for total-tribe-v2.

Wraps Meta's TRIBE v2 model with a single-GPU placement strategy and a
file-hash-keyed prediction cache. Image inputs are converted to short video
clips client-side before being routed through the video pipeline.

This module is a STUB. The real model load is gated on Meta approving
LLaMA-3.2-3B access; see docs/RISKS.md.
"""

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Modalities accepted at the inference layer. Image arrives as a video clip
# after client-side conversion in `_image_to_video`.
_REAL_MODALITIES = frozenset({"text", "audio", "video"})
_UI_MODALITIES = frozenset({"image", "text", "audio", "video"})


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, hex string. Used as cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.cache
def _cached_predict(file_hash: str, modality: str, path_str: str) -> tuple[np.ndarray, tuple]:
    """Module-level cache keyed on (file_hash, modality).

    Implemented as a free function (not a method) so it does not retain a
    reference to a ``TribeInference`` instance — that would prevent GC of the
    underlying model. See docs/SPIKE_FINDINGS.md §8.

    TODO(LLaMA-approval): replace stub body with the real predict call.
    """
    logger.warning(
        "STUB _cached_predict called for hash=%s modality=%s — returning zeros.",
        file_hash[:8],
        modality,
    )
    # placeholder shape (T=30, V=20484) matches fsaverage5 vertex count at 1 Hz
    return np.zeros((30, 20484), dtype=np.float32), ()


class TribeInference:
    """Thin wrapper around upstream ``TribeModel``.

    The model is held on a single CUDA device (``cuda:0``); per-encoder
    placement is not supported by the upstream API. See ADR-0002.
    """

    def __init__(
        self,
        enable_text: bool,
        cache_dir: Path,
        device: str = "cuda:0",
    ) -> None:
        self.enable_text = enable_text
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._model: object | None = None
        logger.warning(
            "TribeInference initialised in STUB mode "
            "(enable_text=%s, cache_dir=%s, device=%s). "
            "Real model load is gated on LLaMA-3.2-3B approval.",
            enable_text,
            self.cache_dir,
            self.device,
        )

    def _load(self) -> None:
        """Load the upstream model onto ``self.device``.

        TODO(LLaMA-approval): this will call
        ``TribeModel.from_pretrained("facebook/tribev2", cache_folder=..., device=...)``
        and assign to ``self._model``.
        """
        raise NotImplementedError(
            "Model loading is gated on LLaMA-3.2-3B approval; see "
            "CHANGELOG.md and docs/RISKS.md."
        )

    def predict(
        self,
        modality: str,
        file_path: Path,
    ) -> tuple[np.ndarray, tuple]:
        """Run inference for a single input file.

        Returns ``(preds, segments)`` where ``preds`` has shape ``(T, 20484)``.

        TODO(LLaMA-approval): replace stub body with the real call after
        ``_load()`` is implemented.
        """
        if modality not in _UI_MODALITIES:
            raise ValueError(
                f"unknown modality {modality!r}; expected one of {sorted(_UI_MODALITIES)}"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if modality == "image":
            path = self._image_to_video(path)
            modality = "video"

        if modality == "text" and not self.enable_text:
            raise RuntimeError("text input requested but ENABLE_TEXT=false; see .env.example")

        return _cached_predict(_hash_file(path), modality, str(path))

    def _image_to_video(
        self,
        image_path: Path,
        *,
        duration: float = 5.0,
        fps: int = 4,
    ) -> Path:
        """Replicate a still image into a short video clip via MoviePy.

        Output: a temporary ``.mp4`` of ``duration`` seconds at ``fps`` frames
        per second. See ADR-0005.

        TODO(LLaMA-approval): implement with ``moviepy.editor.ImageClip``.
        """
        raise NotImplementedError(
            "_image_to_video is a stub; MoviePy implementation lands with the "
            "first real inference pass. See ADR-0005."
        )
