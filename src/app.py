"""Gradio UI for total-tribe-v2.

Four input tabs (image / audio / text / video) → ``TribeInference`` →
``RegionInterpreter`` → brain map + top-K region table + time slider.

This file is a STUB. Real inference is wired up after LLaMA-3.2-3B approval;
see docs/RISKS.md and the ``# TODO(LLaMA-approval):`` block below.

Single-user constraint: the module-level ``LAST`` dict caches the most-recent
prediction so the slider can re-render without re-running inference. This
intentionally does not generalise to multi-tenant; see ADR-0003.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from . import inference, interpretation

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("TRIBE_CACHE", "/app/cache"))
ATLAS_DIR = Path("atlases")
ENABLE_TEXT = os.environ.get("ENABLE_TEXT", "true").lower() in {"1", "true", "yes"}

# Module-level state: most-recent prediction. Single-user only.
LAST: dict[str, Any] = {"preds": None, "segments": None, "modality": None}


def _placeholder_brain_html(t: int) -> str:
    """Stub renderer until nilearn integration lands."""
    return (
        "<div style='padding:1em;border:1px dashed #888;'>"
        f"<b>Brain view placeholder</b><br>"
        f"Time index: <code>{t}</code><br>"
        "Real rendering wired up post-LLaMA approval."
        "</div>"
    )


def run_inference(
    modality: str,
    image: str | None,
    audio: str | None,
    text: str | None,
    video: str | None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Top-level Gradio callback.

    Routes the active modality's path to ``TribeInference.predict``, caches
    the result in ``LAST``, and returns ``(brain_html, region_rows, max_t)``.

    TODO(LLaMA-approval): once the real model is wired, this will call
    ``_engine.predict(modality, path)`` and feed the result through
    ``_interpreter.top_regions``. For now it returns a placeholder so the UI
    layout is verifiable without GPU.
    """
    path_map = {"image": image, "audio": audio, "text": text, "video": video}
    path = path_map.get(modality)
    if not path:
        return _placeholder_brain_html(0), [], 0

    # TODO(LLaMA-approval): replace with real engine + interpreter calls.
    LAST["modality"] = modality
    LAST["preds"] = None
    LAST["segments"] = None
    logger.info("Stub inference: modality=%s path=%s", modality, path)
    return _placeholder_brain_html(0), [], 0


def render_at(t: int) -> str:
    """Re-render the brain view at time index ``t`` using ``LAST``.

    TODO(LLaMA-approval): plug ``_interpreter.top_regions(LAST['preds'][t])``
    + nilearn surface render for both hemispheres.
    """
    return _placeholder_brain_html(t)


def build_ui() -> Any:
    """Construct and return the Gradio Blocks app (not launched)."""
    import gradio as gr  # heavy import; defer until runtime

    with gr.Blocks(title="total-tribe-v2") as demo:
        gr.Markdown("# total-tribe-v2\nBrain-encoding predictions on fsaverage5.")

        with gr.Tabs():
            with gr.Tab("image"):
                image_in = gr.Image(
                    type="filepath",
                    label="Still image (converted to a 5 s video clip)",
                )
            with gr.Tab("audio"):
                audio_in = gr.Audio(type="filepath", label="Audio (wav / mp3 / flac / ogg)")
            with gr.Tab("text"):
                text_in = gr.Textbox(
                    label="Text",
                    placeholder="Paste a passage (≤30 s of reading) …",
                    lines=8,
                )
            with gr.Tab("video"):
                video_in = gr.Video(label="Video (mp4 / avi / mkv / mov / webm)")

        with gr.Row():
            modality = gr.Radio(
                choices=["image", "audio", "text", "video"],
                value="video",
                label="Active modality",
            )
            run_btn = gr.Button("Run inference", variant="primary")

        brain_view = gr.HTML(label="Brain (LH + RH)")
        region_table = gr.JSON(label="Top regions")
        slider = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            label="Time (s) — BOLD response peaks ~5 s after stimulus onset",
        )

        run_btn.click(
            fn=run_inference,
            inputs=[modality, image_in, audio_in, text_in, video_in],
            outputs=[brain_view, region_table, slider],
        )
        slider.change(fn=render_at, inputs=[slider], outputs=[brain_view])

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Starting total-tribe-v2 (ENABLE_TEXT=%s, CACHE_DIR=%s)",
        ENABLE_TEXT,
        CACHE_DIR,
    )
    # The real inference engine + interpreter live here. They are instantiated
    # lazily so this module can be imported in CPU-only tests.
    _engine = inference.TribeInference(  # noqa: F841 (used in real call below)
        enable_text=ENABLE_TEXT,
        cache_dir=CACHE_DIR,
    )
    _interpreter = interpretation.RegionInterpreter(atlas_dir=ATLAS_DIR)  # noqa: F841
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
