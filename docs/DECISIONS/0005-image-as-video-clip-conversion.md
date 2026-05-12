# ADR-0005: Convert still images to a short video clip client-side; route through video pipeline

- **Status:** Accepted
- **Date:** 2026-05-11

## Context

The UI exposes four input tabs (image, audio, text, video) because that is what users expect. The published TRIBE v2 checkpoint, however, has only **three** real input modalities. Spike findings (see `docs/SPIKE_FINDINGS.md` §1, §3) confirmed:

- `model_build_args.feature_dims` contains exactly `{'audio', 'text', 'video'}` — there is no `image` key.
- The `state_dict` has projectors for `text`, `audio`, and `video` only.
- The `image_feature` entry that appears in `config.yaml` is a misleading training-time artifact: its extractor class is `HuggingFaceVideo` (the same as `video_feature`), and `from_pretrained` explicitly pops `data.image_feature.infra` from the config during construction, disabling it. Its output is not fed into the fusion transformer.
- `get_events_dataframe` accepts only `text_path`, `audio_path`, `video_path` — no `image_path`.
- The `CreateVideosFromImages` event transform exists in `eventstransforms.py` but is **not** in the default `data.study.transforms` list, so feeding raw `type="Image"` events to a stock-loaded model accomplishes nothing.

The user-facing requirement remains: they want an image-input tab. The most honest implementation is to handle the image-to-video conversion ourselves, transparently, before the request hits the model.

## Decision

The image input tab is implemented by **client-side conversion to a short video clip**: a single still image is replicated into a ~5 s clip at ~4 fps using MoviePy, then routed through exactly the same code path that handles uploaded video. The user does not need to know about this; the UI surfaces an image input and gets a video-quality prediction back.

VRAM cost of image input equals video input, because both paths feed V-JEPA2-ViTg. The 30 s input cap from [ADR-0002](./0002-single-gpu-placement.md) applies; a 5 s clip is well inside it.

## Consequences

**Positive:**

- Four-tab UI matches user expectations without lying about model capabilities.
- Single inference path. The video pipeline is the only thing that ever sees pixels — no special-casing in `TribeInference`.
- Predicts cleanly on face/scene/object close-ups, which is the dominant image use case for a brain-encoding demo.

**Negative:**

- The replicated-frame clip is not a real video. V-JEPA2's temporal prior expects motion; static replication may look weird to the encoder. Tracked as a verification risk in `docs/RISKS.md` ("`CreateVideosFromImages` quality for V-JEPA2"). Mitigated by a deploy-time smoke test with a face close-up (FFA should activate at t≈5 s).
- One image input == one full V-JEPA2 forward pass on a video tensor. Slightly wasteful compared to a hypothetical image-only encoder, but the wasted compute is bounded by the 5 s clip and is well within the 24 GB ceiling.

## Alternatives considered

- **Drop the image input tab entirely.** Rejected — user-requested UX. Image input is the most intuitive entry point for a non-neuroscience audience ("what does the brain do when it sees a face?").
- **Use `CreateVideosFromImages` server-side as an event-transform.** Rejected — it is not in the default `data.study.transforms` list and is undocumented for inference. Wiring it in would require monkey-patching the upstream `data.study.transforms` list at construction time, which is fragile against upstream releases.
- **Request a dedicated image encoder upstream.** Rejected — out of scope for v1. If/when Meta publishes an image-projector variant, we revisit with a follow-up ADR.
