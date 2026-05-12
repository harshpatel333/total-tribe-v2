"""Unit tests for ``src.inference``.

These tests exercise the stub paths only -- the upstream model load is gated on
LLaMA-3.2-3B approval, so ``TribeInference._load()`` is expected to raise
``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import inference


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


def test_load_raises_until_llama_approved(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    with pytest.raises(NotImplementedError, match="LLaMA"):
        engine._load()


def test_predict_unknown_modality_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    bogus = tmp_path / "x.bin"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown modality"):
        engine.predict("not-a-modality", bogus)


def test_predict_missing_file_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.predict("audio", tmp_path / "nope.wav")


def test_predict_text_with_text_disabled_raises(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=False, cache_dir=tmp_path)
    p = tmp_path / "x.txt"
    p.write_text("a passage")
    with pytest.raises(RuntimeError, match="ENABLE_TEXT"):
        engine.predict("text", p)


def test_predict_image_routes_via_video_pipeline(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    # _image_to_video is a stub that raises NotImplementedError; predict()
    # must surface that.
    with pytest.raises(NotImplementedError, match="_image_to_video"):
        engine.predict("image", p)


def test_predict_audio_returns_stub_zeros(tmp_path: Path) -> None:
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    preds, segments = engine.predict("audio", p)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (30, 20484)
    assert preds.dtype == np.float32
    assert (preds == 0).all()
    assert segments == ()


def test_cached_predict_is_idempotent(tmp_path: Path) -> None:
    """Same (hash, modality) should return the same object from the cache."""
    engine = inference.TribeInference(enable_text=True, cache_dir=tmp_path)
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    a, _ = engine.predict("audio", p)
    b, _ = engine.predict("audio", p)
    assert a is b  # cached free-function returns same array object
