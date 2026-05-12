# Spike findings — corrections to the original TECH_SPEC

**Date:** 2026-05-11
**Spike performed by:** verifying upstream API + loading the published TRIBE v2 checkpoint structure on a 3× RTX 3090 devcontainer.
**Verified by:** Harsh Patel (account: harshpatel333)

The original TECH_SPEC was drafted before the upstream API was inspected. Several load-bearing claims turned out to be wrong. This document is the source of truth for what differs from the spec. The revised `docs/TECH_SPEC.md` (v2) will reflect all of these.

---

## What changed

### 1. The model has 3 modalities, not 4
- Checkpoint's `model_build_args.feature_dims = {'audio': (2, 1024), 'text': (2, 3072), 'video': (2, 1408)}`
- `state_dict` has projectors for `text`, `audio`, `video` only — none for image
- `image_feature` in config.yaml is a misleading artifact: its extractor class is `HuggingFaceVideo` (same as `video_feature`), it consumes Video events (not Image events), and `from_pretrained` explicitly pops `data.image_feature.infra` from config (disabling its infra). Output isn't fed into the fusion. Likely a training-time artifact.

### 2. There is no per-encoder GPU placement
- `TribeModel` exposes NO per-encoder attributes (no `model.text_encoder`, `model.audio_encoder`, `model.video_encoder`, `model.fusion_transformer`). The original spec's §6.1 pseudocode is invalid.
- The actual model is a monolithic `nn.Module` under `model._model`, built from `xp.brain_model_config.build(**build_args)`.
- `from_pretrained` accepts `device="auto"` (defaults to cuda if available) or any single device string. Multi-GPU dispatch is not in the public API.
- Upstream FSDP support exists ONLY for training. Inference is single-device.
- **Implication:** the original spec's 3-GPU smallest-on-0/largest-on-2 strategy DOES NOT WORK. v1 must use a single GPU.

### 3. There is no separate image input path
- `get_events_dataframe` accepts only `text_path`, `audio_path`, `video_path` — no `image_path`.
- Raw `type="Image"` events go nowhere in the default pipeline. The `CreateVideosFromImages` transform exists in `eventstransforms.py` but is NOT in the default `data.study.transforms` list.
- **v1 UI handles image input by client-side conversion** (PIL → MoviePy → short video clip, ~5 s @ 4 fps), then calling `get_events_dataframe(video_path=...)`. The user doesn't need to know.
- VRAM cost of image input == VRAM cost of video input (both use V-JEPA2-ViTg).

### 4. Install incantations differ from the spec
- `tribev2` is NOT on PyPI. Install from git: `pip install 'tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git'`.
- **DO NOT pre-pin numpy <2.1.0** — TRIBE 0.1.0 explicitly requires `numpy==2.2.6`. The DataCamp tutorial's advice is stale.
- `huggingface-cli` is deprecated in modern `huggingface_hub` (≥0.40). Use `hf` (e.g. `hf auth login --token ...`, `hf auth whoami`).
- Python 3.12 works; spec said 3.11+. Tested with `cpython-3.12.13` via uv on Ubuntu 22.04.
- `torch==2.6.0+cu124` was selected by uv resolver against the upstream `pyproject.toml`. CUDA driver 13.0 (R580) is forward-compatible.

### 5. VRAM accounting (revised)
| Component | VRAM |
|---|---|
| LLaMA-3.2-3B | ~7 GB |
| V-JEPA2-ViTg | ~14 GB |
| W2v-BERT 2.0 | ~1 GB |
| Fusion + projectors | ~1 GB |
| Working activations / KV / buffers | budget 0–2 GB |
| **Total target on one 3090** | **~23 GB** (tight; 24 GB cap) |

The earlier "28–32 GB" total in the spec was probably correct for "model weights as listed" but didn't acknowledge that they all live on a single device. The DataCamp tutorial's "40 GB minimum" was a conservative single-card recommendation including activations under longer-clip load. Our v1 enforces a 30 s input cap to stay under the 24 GB ceiling.

### 6. UI rendering bug in spec §6.3
`render_brain(t)` returns `lh.get_iframe()` — only the left hemisphere is rendered. RH activations are computed for the region table but never visualized. v1 must render both hemispheres (side-by-side or combined view).

### 7. Module-level `LAST` global in app.py is single-user
Spec §6.3 uses module-level state for predictions to make the slider interactive. This breaks under concurrent users. Fine for v1 (single-user-behind-basic-auth) but document the constraint in CLAUDE.md / TECH_SPEC.

### 8. `lru_cache` on a method leaks
Spec §6.1's `@functools.lru_cache(maxsize=32)` on `_predict_cached(self, file_hash, modality, path)` retains references to `self`, preventing GC of the model. Use a free function or `functools.cache` keyed on the file hash only.

### 9. §8.2 Neurosynth lookup script is pseudocode
The spec sketches NiMARE + MKDAChi2 but doesn't produce a runnable script. Either:
- (a) provide a real `scripts/build_neurosynth_lookup.py` that produces `atlases/parcel_lookup.json`, or
- (b) hand-curate from Glasser et al. 2016 supplementary materials (note: Glasser's supp has anatomical labels, not "Cognitive Function" — spec was inaccurate on this).

v1 ships a hand-curated subset for the most-likely-active 50 parcels, with the auto-generation script committed but optional.

### 10. v1 hardware lock
Single RTX 3090, 24 GB. Other GPU workloads on the box (e.g. Ollama) must be paused or pinned to other devices when TRIBE is serving. Managed manually outside this codebase.

---

## Resolved-decisions diff vs spec §13

| Spec §13 item | Original | Now |
|---|---|---|
| Subdomain | tribev2.ws.coursebite.ai (Dokploy UI) | Unchanged |
| Text input | Enabled by default | Unchanged (`ENABLE_TEXT=true`) |
| Modality count | 3 (text/audio/video) | Unchanged at inference, but UI surfaces 4 tabs with image converted to video client-side |
| GPU strategy | 3-GPU per-encoder placement | **Single GPU (`cuda:0`)** — spec was wrong |
| Numpy pin | `<2.1.0` | **No pre-pin** — TRIBE requires 2.2.6 |
| HF CLI | `huggingface-cli login` | `hf auth login --token ...` |

---

## What still needs verifying

- **VRAM peak under real load** — blocked on Meta approving LLaMA-3.2-3B access for `harshpatel333`. Once approved, run a short video prediction and confirm peak ≤ 24 GB.
- **`get_events_dataframe` actual column set** — we have the docstring claim (`type`, `filepath`, `start`, `duration`, `timeline`, `subject`) but haven't run it. Verify post-approval.
- **`CreateVideosFromImages` interaction with the model's transform pipeline** — confirmed not in default pipeline, but if our client-side conversion produces video clips that look weird to V-JEPA2, the predictions may be noisy. Test with a face close-up at deploy time; FFA should activate at t≈5s.
- **HCP-MMP1 `.annot` files compatibility with fsaverage5** — the standard download is on fsaverage7. We need the fsaverage5-resampled version per spec §8.1. Verify.

Pending items go in `docs/RISKS.md`.
