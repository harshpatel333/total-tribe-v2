"""Inference engine for total-tribe-v2.

Wraps Meta's TRIBE v2 model with a single-GPU placement strategy and a
file-hash-keyed prediction cache. Image inputs are converted to short video
clips before being routed through the video pipeline.

Bimodal v1: when ``enable_text=False`` the wrapper passes
``config_update={"data.text_feature": None}`` to ``TribeModel.from_pretrained``
so the LLaMA-3.2-3B text encoder is never instantiated. Flip ``ENABLE_TEXT``
to enable text once Meta approval lands; no code change required.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Modalities exposed at the UI layer. Image is routed through the video
# pipeline after client-side conversion by `_image_to_video`.
_UI_MODALITIES = frozenset({"image", "text", "audio", "video"})

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

Modality = Literal["text", "audio", "video", "image"]


_PREDICT_CACHE: dict[tuple[str, str], tuple[np.ndarray, list]] = {}


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, hex string. Used as cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_predict(
    model: object,
    modality: str,
    file_path: Path,
) -> tuple[np.ndarray, list]:
    """Run ``model.get_events_dataframe`` + ``model.predict`` with a file-hash cache.

    The cache is module-level (not bound to ``self``) so the ``TribeInference``
    instance — and therefore the GPU-resident model — can be garbage collected
    without being pinned by ``functools.lru_cache``.

    Args:
        model: an upstream ``TribeModel`` instance with ``get_events_dataframe``
            and ``predict`` methods.
        modality: one of ``"text"``, ``"audio"``, ``"video"``. (Image is
            converted to video upstream.)
        file_path: path to the stimulus file. Must exist.

    Returns:
        Tuple ``(preds, segments)`` where ``preds`` has shape ``(T, 20484)``.
    """
    key = (_hash_file(file_path), modality)
    cached = _PREDICT_CACHE.get(key)
    if cached is not None:
        logger.info("predict cache hit for hash=%s modality=%s", key[0][:8], modality)
        return cached

    kw = {f"{modality}_path": str(file_path)}
    df = model.get_events_dataframe(**kw)  # type: ignore[attr-defined]
    preds, segments = model.predict(events=df, verbose=False)  # type: ignore[attr-defined]
    preds_np = np.asarray(preds)
    result = (preds_np, list(segments))
    _PREDICT_CACHE[key] = result
    logger.info(
        "predict cached hash=%s modality=%s shape=%s segments=%d",
        key[0][:8],
        modality,
        preds_np.shape,
        len(result[1]),
    )
    return result


class TribeInference:
    """Thin wrapper around upstream ``TribeModel``.

    The model is held on a single CUDA device (``cuda:0``); per-encoder
    placement is not supported by the upstream API. See ADR-0002.

    Args:
        enable_text: when ``False`` the text encoder is disabled via
            ``config_update={"data.text_feature": None}``. When ``True``
            the full trimodal model is loaded (requires LLaMA-3.2-3B access).
        cache_dir: HuggingFace cache directory passed as ``cache_folder``.
        device: torch device string. Defaults to ``"cuda:0"``.
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
        logger.info(
            "TribeInference initialised (enable_text=%s, cache_dir=%s, device=%s)",
            enable_text,
            self.cache_dir,
            self.device,
        )

    def _load(self) -> None:
        """Load the upstream TRIBE v2 model onto ``self.device``. Idempotent.

        Sets ``self._model``. Skips text-encoder instantiation when
        ``enable_text=False``.
        """
        if self._model is not None:
            return

        from tribev2 import TribeModel

        # Bimodal mode: disable the text encoder by nulling its config slot AND
        # dropping "text" from features_to_use. The latter is required because
        # tribev2.main.Data.get_loaders iterates features_to_use and calls
        # `getattr(self, f"{modality}_feature")`; with "text" still present that
        # returns None and crashes inside the data pipeline.
        config_update: dict[str, object] | None
        if self.enable_text:
            config_update = None
        else:
            config_update = {
                "data.text_feature": None,
                "data.features_to_use": ["audio", "video"],
            }
        logger.info(
            "loading TribeModel (config_update=%s, device=%s)",
            config_update,
            self.device,
        )
        self._model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=str(self.cache_dir),
            device=self.device,
            config_update=config_update,
        )
        mode = "trimodal: text+audio+video" if self.enable_text else "bimodal: audio+video"
        logger.info("loaded %s", mode)

    def predict(
        self,
        modality: Modality,
        file_path: Path,
    ) -> tuple[np.ndarray, list]:
        """Run inference for a single input file.

        Args:
            modality: one of ``"text"``, ``"audio"``, ``"video"``, ``"image"``.
                Image is converted to a 5 s silent video clip and dispatched
                through the video pipeline.
            file_path: path to the stimulus file. Must exist.

        Returns:
            Tuple ``(preds, segments)``. ``preds`` is a ``np.ndarray`` of shape
            ``(T, 20484)`` (one row per second on fsaverage5). ``segments`` is
            the segmentation list returned by the upstream model.

        Raises:
            ValueError: unknown modality.
            FileNotFoundError: the stimulus file does not exist.
            RuntimeError: ``modality="text"`` with ``enable_text=False``.
        """
        if modality not in _UI_MODALITIES:
            raise ValueError(
                f"unknown modality {modality!r}; expected one of {sorted(_UI_MODALITIES)}"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if modality == "text" and not self.enable_text:
            raise RuntimeError(
                "text disabled; set ENABLE_TEXT=true and obtain Meta LLaMA-3.2-3B access"
            )

        self._load()
        assert self._model is not None  # for type checkers; _load guarantees this

        if modality == "image":
            video_path = self._image_to_video(path)
            return _cached_predict(self._model, "video", video_path)

        return _cached_predict(self._model, modality, path)

    def _image_to_video(
        self,
        image_path: Path,
        *,
        duration: float = 5.0,
        fps: int = 4,
    ) -> Path:
        """Replicate a still image into a short silent video clip via MoviePy.

        Args:
            image_path: source image. Suffix must be one of
                ``.jpg``, ``.jpeg``, ``.png``, ``.webp``.
            duration: clip duration in seconds.
            fps: frames per second for the encoded mp4.

        Returns:
            Path to a temp ``.mp4``. The caller is responsible for cleanup,
            but the file is created in the OS temp dir so it will be reaped
            on container restart.

        Raises:
            ValueError: unsupported image extension.
        """
        suffix = image_path.suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            raise ValueError(
                f"unsupported image extension {suffix!r}; "
                f"expected one of {sorted(_IMAGE_SUFFIXES)}"
            )

        # moviepy 2.x: fluent API uses with_duration / with_fps; v1's set_duration
        # was renamed. We pin moviepy>=1.0 in pyproject; 2.x is the installed
        # version under [gpu].
        from moviepy import ImageClip

        clip = ImageClip(str(image_path)).with_duration(duration).with_fps(fps)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = Path(tmp.name)
        clip.write_videofile(
            str(out_path),
            codec="libx264",
            audio=False,
            logger=None,
        )
        clip.close()
        logger.info("image -> video clip: %s (%.1fs @ %d fps)", out_path, duration, fps)
        return out_path
