"""Unit tests for ``src.inference``.

CPU-only. The real ``TribeModel.from_pretrained`` call is GPU-bound and gated
on Meta approval for text; these tests inject a fake model so the dispatch,
caching, and image-to-video paths can be exercised on commodity hardware.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import inference


@pytest.fixture(autouse=True)
def _reset_predict_cache() -> None:
    """Each test starts with a clean module-level predict cache."""
    inference._PREDICT_CACHE.clear()


def _fake_model(t: int = 30) -> MagicMock:
    """Build a fake TribeModel with deterministic predict output."""
    m = MagicMock()
    m.get_events_dataframe.return_value = pd.DataFrame({"event": ["x"]})
    m.predict.return_value = (np.ones((t, 20484), dtype=np.float32), ["seg0"])
    return m


def test_hash_file_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = inference._hash_file(p)
    h2 = inference._hash_file(p)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_file_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = inference._hash_file(p)
    p.write_bytes(b"hello world!")
    h2 = inference._hash_file(p)
    assert h1 != h2


def test_tribe_inference_init_does_not_load_model(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    assert engine._model is None
    assert engine.enable_text is True
    assert engine.device == "cuda:0"


def test_predict_unknown_modality_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    bogus = tmp_path / "x.bin"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown modality"):
        engine.predict("not-a-modality", bogus)  # type: ignore[arg-type]


def test_predict_missing_file_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.predict("audio", tmp_path / "nope.wav")


def test_predict_text_with_text_disabled_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    p = tmp_path / "x.txt"
    p.write_text("a passage")
    with pytest.raises(RuntimeError, match="text disabled"):
        engine.predict("text", p)


def test_predict_audio_dispatches_to_model(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    engine._model = _fake_model(t=30)
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    preds, segments = engine.predict("audio", p)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (30, 20484)
    assert segments == ["seg0"]
    engine._model.get_events_dataframe.assert_called_once_with(audio_path=str(p))
    engine._model.predict.assert_called_once()


def test_predict_video_dispatches_to_model(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    engine._model = _fake_model(t=10)
    p = tmp_path / "x.mp4"
    p.write_bytes(b"\x00" * 32)

    preds, _ = engine.predict("video", p)
    assert preds.shape == (10, 20484)
    engine._model.get_events_dataframe.assert_called_once_with(video_path=str(p))


def test_cached_predict_is_idempotent(tmp_path: Path) -> None:
    """Same (file hash, modality) hits the cache and avoids re-running predict."""
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    engine._model = _fake_model(t=30)
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    a, _ = engine.predict("audio", p)
    b, _ = engine.predict("audio", p)

    assert a is b  # same numpy array object returned from cache
    assert engine._model.predict.call_count == 1
    assert engine._model.get_events_dataframe.call_count == 1


def test_predict_image_routes_through_video_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``image`` modality call should convert -> mp4 then dispatch video_path."""
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    engine._model = _fake_model(t=5)
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    fake_mp4 = tmp_path / "fake.mp4"
    fake_mp4.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(inference.TribeInference, "_image_to_video", lambda self, p: fake_mp4)

    preds, _ = engine.predict("image", image)
    assert preds.shape == (5, 20484)
    engine._model.get_events_dataframe.assert_called_once_with(video_path=str(fake_mp4))


def test_image_to_video_rejects_unsupported_extension(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    p = tmp_path / "x.bmp"
    p.write_bytes(b"BM" + b"\x00" * 100)
    with pytest.raises(ValueError, match="unsupported image extension"):
        engine._image_to_video(p)
