"""Gradio UI for total-tribe-v2.

Four input tabs (image / audio / text / video) → ``TribeInference`` →
``RegionInterpreter`` → both-hemisphere brain map + top-K region table +
time slider.

Single-user constraint: the module-level ``LAST`` dict caches the most-recent
prediction so the slider can re-render without re-running inference. This
intentionally does not generalise to multi-tenant; see ADR-0003. v1 is
deployed behind basic auth; one user, one GPU.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from . import inference, interpretation

logger = logging.getLogger(__name__)

ENABLE_TEXT = os.environ.get("ENABLE_TEXT", "false").lower() in {"1", "true", "yes"}
CACHE_DIR = Path(os.environ.get("TRIBE_CACHE", "/app/cache"))
ATLAS_DIR = Path(os.environ.get("ATLASES_DIR", "atlases"))

# fsaverage5 has 10242 vertices per hemisphere.
_VERTS_PER_HEMI = 10242
# Hemodynamic lag: BOLD peaks ~5 s after stimulus onset. We seed the slider
# at this offset so the first frame the user sees is the peak response.
_HEMODYNAMIC_LAG_S = 5

# Module-level state: most-recent prediction. Single-user only.
LAST: dict[str, Any] = {"preds": None, "segments": None}

_TEXT_PENDING_HTML = (
    "<p>Text input requires Meta approval for "
    '<a href="https://huggingface.co/meta-llama/Llama-3.2-3B" '
    'target="_blank" rel="noopener noreferrer">Llama-3.2-3B</a>. '
    "Currently pending — try audio, video, or image for now.</p>"
)

_NO_INPUT_HTML = (
    "<p>Upload an image, audio, or video (or paste text once Meta access "
    "lands) and click <b>Predict brain response</b>.</p>"
)

# Lazily instantiated. We can't construct them at import time because tests
# import this module on CPU and don't have HF cache or atlases available.
_engine: inference.TribeInference | None = None
_interpreter: interpretation.RegionInterpreter | None = None
_fsaverage: Any = None


def _get_engine() -> inference.TribeInference:
    global _engine
    if _engine is None:
        _engine = inference.TribeInference(
            enable_text=ENABLE_TEXT,
            cache_dir=CACHE_DIR,
            device="cuda:0",
        )
    return _engine


def _get_interpreter() -> interpretation.RegionInterpreter:
    global _interpreter
    if _interpreter is None:
        _interpreter = interpretation.RegionInterpreter(atlas_dir=ATLAS_DIR)
    return _interpreter


def _get_fsaverage() -> Any:
    global _fsaverage
    if _fsaverage is None:
        from nilearn.datasets import fetch_surf_fsaverage

        _fsaverage = fetch_surf_fsaverage("fsaverage5")
    return _fsaverage


def render_brain(t: int) -> str:
    """Render LH + RH cortical surfaces at timestep ``t`` as side-by-side iframes.

    Uses the **pial** mesh (not inflated) so the surface looks like a real
    brain rather than a balloon. Sulcal depth provides the grey-on-grey
    anatomical context; the ``hot`` colormap overlays a 5% threshold of the
    predicted activation map.
    """
    from nilearn.plotting import view_surf

    preds = LAST.get("preds")
    if preds is None:
        return _NO_INPUT_HTML

    if t < 0 or t >= preds.shape[0]:
        t = max(0, min(t, preds.shape[0] - 1))

    activation = preds[t]
    lh = activation[:_VERTS_PER_HEMI]
    rh = activation[_VERTS_PER_HEMI:]
    fsa = _get_fsaverage()

    # Pial mesh looks anatomically brain-like; inflated mesh looks like a
    # balloon and confuses non-neuroscientists. We trade gyral detail for
    # interpretability.
    lh_view = view_surf(
        surf_mesh=fsa["pial_left"],
        surf_map=lh,
        bg_map=fsa.get("sulc_left"),
        hemi="left",
        threshold="5%",
        cmap="hot",
        symmetric_cmap=False,
        title=f"Left hemisphere — t={t}s",
    )
    rh_view = view_surf(
        surf_mesh=fsa["pial_right"],
        surf_map=rh,
        bg_map=fsa.get("sulc_right"),
        hemi="right",
        threshold="5%",
        cmap="hot",
        symmetric_cmap=False,
        title=f"Right hemisphere — t={t}s",
    )

    lh_html = lh_view.get_iframe()
    rh_html = rh_view.get_iframe()
    return (
        '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:360px;">{lh_html}</div>'
        f'<div style="flex:1;min-width:360px;">{rh_html}</div>'
        "</div>"
    )


def _top_regions_at(t: int) -> list[dict[str, Any]]:
    preds = LAST.get("preds")
    if preds is None:
        return []
    t = max(0, min(t, preds.shape[0] - 1))
    return _get_interpreter().top_regions(preds[t], k=8)


def _summary_markdown(top_regions: list[dict[str, Any]]) -> str:
    """Produce the layman summary Markdown for a given region table."""
    if not top_regions:
        return (
            "*Upload a stimulus and click **Predict brain response** to see "
            "a plain-English interpretation here.*"
        )
    summary = _get_interpreter().summarize(top_regions)
    lines = [f"### {summary['headline']}", "", summary["narrative"]]
    cats = summary.get("categories") or []
    if cats:
        lines.append("")
        lines.append("**Network breakdown** (of the top 8 regions):")
        for c in cats:
            sign = "+" if c["mean_activation"] >= 0 else "−"
            lines.append(
                f"- {c['display']} — {c['count']} region"
                f"{'s' if c['count'] != 1 else ''}, "
                f"mean activation {sign}{abs(c['mean_activation']):.2f}"
            )
    return "\n".join(lines)


def _regions_table(top_regions: list[dict[str, Any]]) -> list[list[Any]]:
    """Format a region list as a friendly tabular DataFrame for Gradio."""
    rows: list[list[Any]] = []
    for r in top_regions:
        rows.append(
            [
                r.get("name", r.get("parcel", "?")),
                f"{r.get('activation', 0.0):+.2f}",
                (r.get("category") or "").replace("_", " "),
                r.get("function", ""),
            ]
        )
    return rows


def _pick_active_input(
    image: str | None,
    audio: str | None,
    text: str | None,
    video: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(modality, value)`` for the single non-empty input, or ``(None, None)``.

    If more than one input is set, prefer the order: image > audio > text > video.
    Empty strings (textbox default) count as missing.
    """
    if image:
        return "image", image
    if audio:
        return "audio", audio
    if text and text.strip():
        return "text", text
    if video:
        return "video", video
    return None, None


def run_inference(
    image: str | None,
    audio: str | None,
    text: str | None,
    video: str | None,
) -> tuple[str, str, list[list[Any]], Any]:
    """Top-level Gradio callback.

    Returns ``(summary_md, brain_html, region_rows, slider_update)`` where the
    slider is seeded at ``t = min(_HEMODYNAMIC_LAG_S, T-1)``.
    """
    import gradio as gr

    modality, value = _pick_active_input(image, audio, text, video)
    if modality is None:
        return _summary_markdown([]), _NO_INPUT_HTML, [], gr.update()

    if modality == "text" and not ENABLE_TEXT:
        return _summary_markdown([]), _TEXT_PENDING_HTML, [], gr.update()

    if modality == "text":
        # Persist the text passage to a temp file so the engine's path-based
        # API can consume it. The hash-cache makes repeat predictions free.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(value or "")
            stim_path = Path(tmp.name)
    else:
        stim_path = Path(value or "")

    logger.info("running inference: modality=%s path=%s", modality, stim_path)
    preds, segments = _get_engine().predict(modality, stim_path)  # type: ignore[arg-type]
    LAST["preds"] = preds
    LAST["segments"] = segments

    t_max = max(0, int(preds.shape[0]) - 1)
    t_init = min(_HEMODYNAMIC_LAG_S, t_max)
    brain_html = render_brain(t_init)
    regions = _top_regions_at(t_init)
    return (
        _summary_markdown(regions),
        brain_html,
        _regions_table(regions),
        gr.update(minimum=0, maximum=t_max, value=t_init, step=1),
    )


def on_slider_change(t: int) -> tuple[str, str, list[list[Any]]]:
    """Re-render summary + brain + region table at timestep ``t`` from cached ``LAST``."""
    regions = _top_regions_at(int(t))
    return _summary_markdown(regions), render_brain(int(t)), _regions_table(regions)


def build_ui() -> Any:
    """Construct and return the Gradio Blocks app (not launched)."""
    import gradio as gr

    text_placeholder = (
        "Paste a passage (≤30 s of reading) …"
        if ENABLE_TEXT
        else "(Text disabled — pending Meta approval for Llama-3.2-3B)"
    )

    with gr.Blocks(title="TRIBE v2 — Brain Response Predictor") as demo:
        gr.Markdown("# TRIBE v2 — Predicted fMRI Brain Activity")
        gr.Markdown(
            "Upload an image, audio, text, or video. Output is the predicted "
            "population-average BOLD activation on fsaverage5 at 1 Hz. "
            "The model accounts for ~5 s hemodynamic lag (peak at +5 s). "
            "Inputs ≤ 30 s recommended."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Image"):
                        image_in = gr.Image(
                            type="filepath",
                            label="Still image → 5 s silent clip",
                        )
                    with gr.TabItem("Audio"):
                        audio_in = gr.Audio(type="filepath", label="Audio")
                    with gr.TabItem("Text"):
                        text_in = gr.Textbox(
                            label="Text",
                            lines=4,
                            interactive=ENABLE_TEXT,
                            placeholder=text_placeholder,
                        )
                        if not ENABLE_TEXT:
                            gr.Markdown(
                                "**Text input requires Meta approval for "
                                "[Llama-3.2-3B]"
                                "(https://huggingface.co/meta-llama/Llama-3.2-3B). "
                                "Currently pending.**"
                            )
                    with gr.TabItem("Video"):
                        video_in = gr.Video(label="Video")
                run_btn = gr.Button("Predict brain response", variant="primary")
                # Maximum is reset by run_inference once we know T. Gradio 6
                # requires minimum < maximum at construction time, so we start
                # at (0, 1) and let the callback widen it.
                t_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    label="Timestep (s after stimulus; +5 s ≈ peak)",
                )
            with gr.Column(scale=2):
                summary_md = gr.Markdown(
                    value=_summary_markdown([]),
                    label="Layman summary",
                )
                brain_view = gr.HTML(value=_NO_INPUT_HTML, label="Cortical surface (LH + RH)")
                regions_tbl = gr.Dataframe(
                    headers=["Region", "Activation", "Network", "What it does"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Top 8 active regions",
                )

        run_btn.click(
            run_inference,
            inputs=[image_in, audio_in, text_in, video_in],
            outputs=[summary_md, brain_view, regions_tbl, t_slider],
        )
        t_slider.change(
            on_slider_change,
            inputs=[t_slider],
            outputs=[summary_md, brain_view, regions_tbl],
        )

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Starting total-tribe-v2 (ENABLE_TEXT=%s, CACHE_DIR=%s, ATLAS_DIR=%s)",
        ENABLE_TEXT,
        CACHE_DIR,
        ATLAS_DIR,
    )
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
