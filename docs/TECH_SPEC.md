# Tech Spec — total-tribe-v2

**Version:** v2 (2026-05-11)

> This is a from-scratch revision of an earlier pre-spike draft. Every load-bearing correction in [`docs/SPIKE_FINDINGS.md`](./SPIKE_FINDINGS.md) is folded in. Read SPIKE_FINDINGS first if you want to understand why a given decision changed; this document is the forward-looking source of truth.

---

## §1 Goal

We wrap Meta's TRIBE v2 brain-encoding foundation model in a self-hosted Gradio UI. The user uploads text, audio, video, or a still image; the system predicts the population-averaged fMRI BOLD activation on the fsaverage5 cortical surface (20484 vertices) at 1 Hz, then renders a brain map alongside a top-K HCP-MMP1 region table annotated with Neurosynth-derived semantic descriptors (e.g. "faces", "language", "auditory").

The deployment target is a single RTX 3090 (24 GB) behind Dokploy basic-auth at `tribev2.ws.coursebite.ai`. The goal is research-grade exploration of a state-of-the-art brain encoder, not production traffic — one user, one GPU, one model resident in memory at a time. We optimize for transparency (the user always sees which parcels lit up and why) and for honest failure modes (clear error messages on duration limits, gated model access, GPU OOM) over throughput.

We do not retrain, fine-tune, or modify the underlying TRIBE v2 weights. Everything in this repo is inference-time glue: input normalization, an image-to-video shim, a region interpretation layer, and a Gradio front-end. Per `docs/SPIKE_FINDINGS.md` §2, the public TRIBE API is monolithic — we do not attempt per-encoder GPU placement.

## §2 Non-goals

- No training or fine-tuning. Inference only.
- No multi-tenant auth. A single user behind Dokploy's basic-auth middleware.
- No individual-subject predictions. TRIBE v2 outputs population averages on fsaverage5; we do not approximate per-subject responses.
- No commercial use. The repository is CC-BY-NC-4.0.
- No clinical use. No diagnostic, therapeutic, or medical-device claims, ever.

## §3 Background

TRIBE v2 is a multimodal brain-encoding transformer published by Meta FAIR. Architecturally it has three modality encoders feeding a fusion transformer that predicts cortical BOLD activity on fsaverage5:

| Encoder | Backbone | Purpose |
|---|---|---|
| Text encoder | LLaMA-3.2-3B | Token-level semantic features |
| Audio encoder | W2v-BERT 2.0 | Acoustic + speech features |
| Video encoder | V-JEPA2-ViTg | Spatiotemporal visual features |
| Fusion | Custom transformer + per-modality projectors | Predict BOLD over 20484 vertices @ 1 Hz |

References:

- Model card: https://huggingface.co/facebook/tribev2
- Repository: https://github.com/facebookresearch/tribev2
- DataCamp tutorial: https://www.datacamp.com/tutorial/tribe-v2

**Three real modalities, four UI tabs.** Per `docs/SPIKE_FINDINGS.md` §1, the checkpoint's `model_build_args.feature_dims` is `{'audio': (2, 1024), 'text': (2, 3072), 'video': (2, 1408)}` — there is no `image` key, no image projector in the `state_dict`, and `from_pretrained` explicitly pops `data.image_feature.infra` from the config at load time. The `image_feature` block in upstream `config.yaml` is an unused training-time artifact whose extractor class is `HuggingFaceVideo` (the same class used for video) and whose output is not wired into fusion. Image input in our UI is therefore a client-side affordance: a still image is replicated into a short video clip and fed through the video path. See §6.1 and ADR-0005.

## §4 Hardware budget

We target a single RTX 3090, 24 GB VRAM. Itemized peak VRAM (per `docs/SPIKE_FINDINGS.md` §5):

| Component | VRAM |
|---|---|
| LLaMA-3.2-3B (text encoder) | ~7 GB |
| V-JEPA2-ViTg (video encoder) | ~14 GB |
| W2v-BERT 2.0 (audio encoder) | ~1 GB |
| Fusion transformer + projectors | ~1 GB |
| Activations + KV + buffers (working set) | 0–2 GB |
| **Total target** | **~23 GB** |

Headroom is thin (~1 GB). We enforce a **30 s input duration cap** to keep the working set bounded under the 24 GB ceiling. Setting `ENABLE_TEXT=false` releases the LLaMA-3.2-3B encoder (~7 GB) and lifts the duration cap proportionally; we document this as the escape valve for users on smaller cards or for longer clips, but the default deployment ships text enabled.

Other GPU workloads on the host (e.g. Ollama) must be paused or pinned to other devices when TRIBE is serving. This is managed manually outside this codebase per `docs/SPIKE_FINDINGS.md` §10.

## §5 Architecture

```
Browser
  │  HTTPS, basic auth
  ▼
Traefik (Dokploy)
  │  :7860 proxy
  ▼
┌──────────────────────────────────────────┐
│ tribev2 container (single GPU cuda:0)    │
│                                          │
│  Gradio UI (src/app.py)                  │
│   ├─ 4 input tabs (image/audio/text/video)│
│   ├─ time slider with +5s BOLD lag note  │
│   └─ module-level LAST state             │
│      ▼                                   │
│  TribeInference (src/inference.py)       │
│   ├─ image → video preprocessor          │
│   ├─ free-fn predict cache (file hash)   │
│   └─ TribeModel.predict()                │
│      ▼                                   │
│  RegionInterpreter (src/interpretation.py)│
│   ├─ HCP-MMP1 .annot → parcel labels     │
│   ├─ parcel_lookup.json → Neurosynth terms│
│   └─ top-K aggregator                    │
│      ▼                                   │
│  nilearn surface render (LH + RH)        │
└──────────────────────────────────────────┘
       │
       ▼
Volumes: /app/cache (HF + tribev2 weights), /app/uploads (user files)
```

Traffic flows browser → Traefik (TLS termination + basic-auth) → the single `tribev2` container on port 7860. The container holds the entire TRIBE v2 model in resident GPU memory on `cuda:0` for the lifetime of the process. There is no autoscaling; the deployment is single-replica by design.

## §6 Component breakdown

### §6.1 Inference engine — `src/inference.py`

Class signature stub:

```python
class TribeInference:
    def __init__(
        self,
        enable_text: bool,
        cache_dir: Path,
        device: str = "cuda:0",
    ) -> None: ...

    def _load(self) -> None:
        # TODO(LLaMA-approval): calls TribeModel.from_pretrained
        raise NotImplementedError("gated on LLaMA-3.2-3B approval")

    def predict(self, modality: str, file_path: Path) -> tuple[np.ndarray, list]: ...

    def _image_to_video(
        self,
        image_path: Path,
        *,
        duration: float = 5.0,
        fps: int = 4,
    ) -> Path: ...
```

**Loading.** `_load` calls `TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir, device=device)`. Per `docs/SPIKE_FINDINGS.md` §2, `TribeModel` exposes no per-encoder attributes — there is no `model.text_encoder` or `model.video_encoder` to relocate. The whole model lives on a single device. We pass `device="cuda:0"` and accept it. Multi-GPU dispatch is not part of the public TRIBE API; FSDP exists upstream but only for training. ADR-0002 records this lock.

**Gating.** `_load` raises `NotImplementedError("gated on LLaMA-3.2-3B approval")` until Meta approves the user's LLaMA-3.2-3B access. Once approval is granted, the `# TODO(LLaMA-approval)` marker is removed and the body is filled in. We do not stub the model with a random tensor — we want loud failures, not silent wrong answers.

**Prediction call.** The verified upstream API is:

```python
df = model.get_events_dataframe(
    text_path=None,
    audio_path=None,
    video_path=None,
)
preds, segments = model.predict(events=df, verbose=False)
# preds: np.ndarray, shape (T, 20484) on fsaverage5 at 1 Hz
```

`get_events_dataframe` accepts only `text_path`, `audio_path`, `video_path` (no `image_path` — see §3 and `docs/SPIKE_FINDINGS.md` §3).

**Dispatch by modality:**

| Modality | Path |
|---|---|
| `"image"` | `_image_to_video(path)` → `get_events_dataframe(video_path=tmp_video)` |
| `"audio"` | `get_events_dataframe(audio_path=path)` |
| `"text"`  | `get_events_dataframe(text_path=path)` — requires `ENABLE_TEXT=true`, `HF_TOKEN`, and approved LLaMA access |
| `"video"` | `get_events_dataframe(video_path=path)` |

**Image → video conversion.** `_image_to_video` uses MoviePy to replicate a single still into a ~5 s clip at 4 fps and writes it to a temp file (`tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)`). The returned path is fed into the video pipeline. VRAM cost of image input equals VRAM cost of video input — both hit V-JEPA2-ViTg. See ADR-0005.

**Caching.** Prediction results are memoized via a **module-level free function** decorated with `functools.cache`, keyed on `(sha256_of_file, modality)`. Per `docs/SPIKE_FINDINGS.md` §8, decorating a method with `@functools.lru_cache` retains a reference to `self`, which prevents garbage collection of `TribeInference` and therefore of the multi-GB model. We use a free-function cache instead:

```python
@functools.cache
def _cached_predict(
    file_hash: str,
    modality: str,
    *,
    _runner: "TribeInference",
    file_path: Path,
) -> tuple[np.ndarray, list]:
    return _runner._predict_uncached(modality, file_path)
```

`_runner` is passed as a keyword-only argument and is not part of the cache key (its identity is implicit in the process). `_hash_file(path)` is a free function returning `hashlib.sha256(path.read_bytes()).hexdigest()`.

References: `docs/DECISIONS/0002-single-gpu-placement.md`, `docs/DECISIONS/0005-image-as-video-clip-conversion.md`.

### §6.2 Interpretation layer — `src/interpretation.py`

The interpretation layer turns a `(T, 20484)` prediction tensor into a human-readable region table.

**Inputs.**

- `atlases/lh.HCP-MMP1.annot` and `atlases/rh.HCP-MMP1.annot` — HCP-MMP1 parcellation resampled to fsaverage5 (10242 vertices per hemisphere). Loaded once via `nibabel.freesurfer.read_annot`.
- `atlases/parcel_lookup.json` — hand-curated mapping from parcel label (e.g. `L_FFC`) to a list of Neurosynth-derived semantic terms (e.g. `["faces", "face perception"]`). Ships with ~50 parcels covering the most-likely-active regions; missing parcels yield an empty term list rather than an error.

**API.**

```python
class RegionInterpreter:
    def __init__(
        self,
        lh_annot: Path,
        rh_annot: Path,
        lookup: Path,
    ) -> None: ...

    def top_regions(
        self,
        pred_t: np.ndarray,
        k: int = 8,
    ) -> list[dict]: ...
```

`top_regions(pred_t, k=8)` algorithm:

1. Split `pred_t` (length 20484) into LH (`pred_t[:10242]`) and RH (`pred_t[10242:]`).
2. For each parcel in each hemisphere, compute the mean activation across the vertices belonging to that parcel.
3. Concatenate LH + RH parcel means, sort by absolute activation descending, take the top-K.
4. Return a list of dicts: `{"parcel": "L_FFC", "activation": float, "terms": ["faces"]}`. `terms` is the empty list when the parcel is not in `parcel_lookup.json`.

The layer is pure NumPy + nibabel — no GPU, no torch.

### §6.3 Gradio UI — `src/app.py`

Four input tabs, one output panel.

**Input tabs.**

| Tab | Widget | Notes |
|---|---|---|
| Image | `gr.Image(type="filepath")` | Converted client-side via `_image_to_video` (5 s @ 4 fps). |
| Audio | `gr.Audio(type="filepath")` | Accepts wav, mp3, flac, ogg. |
| Text  | `gr.Textbox` | Writes the contents to a temp `.txt`. Gated on `ENABLE_TEXT=true`; disabled with a banner otherwise. |
| Video | `gr.Video()` | Accepts mp4, avi, mkv, mov, webm. |

**Output panel.**

- **Brain visualization** — `gr.HTML` embedding a nilearn iframe. **Both hemispheres are rendered**, side-by-side or in a combined view. Per `docs/SPIKE_FINDINGS.md` §6, the pre-spike spec rendered only the left hemisphere — that bug is fixed here.
- **Top-K region table** — Markdown or `gr.DataFrame` showing `parcel`, `activation`, `terms`.
- **Time slider** — `gr.Slider(minimum=0, maximum=T-1, step=1, label="Time (s, stimulus onset)")`. The label explicitly notes: "BOLD response peaks ~5 s after stimulus onset; slider time = stimulus time." See `docs/SPIKE_FINDINGS.md` §11 (table row on BOLD lag).

**State management.** A module-level dict `LAST` caches the most recent `(preds, segments, modality, file_hash)` tuple. The slider callback reads `LAST["preds"]` and re-renders without re-running inference. This is the cheap mechanism for an interactive time scrubber.

**Single-user constraint.** Module-level `LAST` is process-global state — it breaks under concurrent users (two simultaneous predictions race on the same dict). We accept this trade-off because the deployment is single-user behind Dokploy basic-auth. Per `docs/SPIKE_FINDINGS.md` §7, this is documented in `CLAUDE.md` and called out inline at the `LAST` declaration. If we ever go multi-user, the fix is per-session state via `gr.State`, not a more elaborate global.

**Gated bodies.** The Gradio callback that wires "Submit" to `TribeInference.predict` is wrapped in a `# TODO(LLaMA-approval):` block. Until approval lands, the callback raises a friendly `gr.Error("LLaMA-3.2-3B approval pending; see SECURITY.md.")` instead of crashing the page.

## §7 Container setup

- **Base image:** `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. CUDA driver 13.0 (R580) is forward-compatible with the 12.4 toolkit per `docs/SPIKE_FINDINGS.md` §4.
- **Python:** 3.12 installed via `apt`, plus `uv` for dependency resolution and venv management.
- **Dependencies:** declared in `pyproject.toml`. `uv pip install -e ".[gpu]"` pulls `tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git` and `torch==2.6.0`. We do not ship a `requirements.txt`.
- **Numpy:** no pre-pin. TRIBE 0.1.0 requires `numpy==2.2.6` and the resolver picks it. The pre-spike spec's `numpy<2.1.0` directive is wrong per `docs/SPIKE_FINDINGS.md` §4.
- **Hugging Face CLI:** modern `hf` (e.g. `hf auth login --token ...`). `huggingface-cli` is deprecated in `huggingface_hub` ≥ 0.40.
- **Compose:** `docker-compose.yml` defines a single service `tribev2` with `device_ids: ["0"]` reservation. **No Traefik labels are written into the compose file** — Dokploy injects them at deploy time. See ADR-0003.

Volumes:

- `/app/cache` — HF + `tribev2` weights. Persistent across deploys.
- `/app/uploads` — temp user files (cleared periodically).

## §8 Atlas + Neurosynth lookup

- Atlases live under `atlases/`. The fsaverage5-resampled `lh.HCP-MMP1.annot` and `rh.HCP-MMP1.annot` files are large and gitignored.
- `scripts/fetch_atlases.py` fetches them idempotently from the Figshare mirror (https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446). Re-running is a no-op if the SHA-256 matches.
- `scripts/build_neurosynth_lookup.py` produces `atlases/parcel_lookup.json` via NiMARE + Neurosynth as a build-time, offline tool. It is committed but optional; the user does not run it at inference time.
- v1 ships a hand-curated `atlases/parcel_lookup.json` covering ~50 well-known parcels — `L_FFC`/`R_FFC` → `["faces"]`, `L_V1`/`R_V1` → `["vision", "visual"]`, `L_PT`/`R_PT` → `["auditory", "speech"]`, `L_44`/`R_44` → `["language", "Broca"]`, etc. The hand-curated set is the source of truth at runtime; the auto-generator is for future expansion.
- Per `docs/SPIKE_FINDINGS.md` §9, Glasser et al. 2016 supplementary materials provide **anatomical** labels, not "cognitive function" labels. Cognitive terms come from Neurosynth (auto-generation) or human judgment (hand-curated). The pre-spike spec conflated these.

## §9 Deployment runbook

See `docs/DEPLOYMENT.md` for the full Dokploy UI workflow.

**Smoke test.** Upload a face close-up video clip → expect `L_FFC` and/or `R_FFC` in the top-K regions at t ≈ 5 s (BOLD lag). If FFA does not light up on a face stimulus, something is wrong (likely the image-to-video conversion produced output V-JEPA2 can't parse, or the atlas mapping is broken). This is the canonical end-to-end correctness check.

## §9b Claude Code automation playbook

When MCP Dokploy tools are available, the deploy sequence is:

1. `mcp__dokploy__project_create` — name `total-tribe-v2`, type `docker-compose`, git repo URL.
2. `mcp__dokploy__app_set_env` — `HF_TOKEN`, `ENABLE_TEXT=true`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASS`.
3. `mcp__dokploy__domain_add` — `tribev2.ws.coursebite.ai`, port 7860, enable Let's Encrypt.
4. `mcp__dokploy__middleware_basicauth_add` — bind to the domain.
5. `mcp__dokploy__app_deploy` — trigger the build.
6. Poll `mcp__dokploy__app_status` until healthy, then HTTP GET `https://tribev2.ws.coursebite.ai/` with basic auth credentials. A 200 response with the Gradio HTML confirms the deployment.

If the MCP tools are unavailable, fall back to the manual Dokploy UI workflow documented in `docs/DEPLOYMENT.md`. The steps map one-to-one.

## §10 OSS hygiene

We cross-reference rather than duplicate. The canonical files:

- `README.md` — quickstart, hardware requirements, badges.
- `CHANGELOG.md` — Keep-a-Changelog format.
- `CONTRIBUTING.md` — dev setup, PR checklist, Conventional Commits.
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1.
- `SECURITY.md` — private security advisory process; 7-day SLA on initial response.
- `CLAUDE.md` — project + behavioral guidelines for Claude Code (single-user constraint, `LAST` global rationale, etc.).
- `docs/DECISIONS/` — ADRs (numbered, immutable, append-only).
- `.github/` — issue and PR templates, CI workflows.

## §11 Known gotchas

| Gotcha | Mitigation |
|---|---|
| 15 s **minimum** input duration (TRIBE temporal stride) | Reject inputs <15 s with a clear UI message |
| Render BOTH hemispheres, not LH only | `RegionInterpreter` returns both; UI renders combined view |
| BOLD response peaks ~5 s after stimulus onset | Label slider explicitly |
| File hash–keyed cache, not method-decorated `lru_cache` | Free-function `@functools.cache` to avoid retaining `self` |
| LLaMA-3.2-3B is gated | `_load()` raises `NotImplementedError` until approval |
| `numpy==2.2.6` required by TRIBE | Do NOT pre-pin numpy |
| `huggingface-cli` deprecated | Use `hf` CLI |
| Single-GPU only at inference | ADR-0002 |

## §12 Future work

- WebGL renderer to replace the nilearn iframe (faster, interactive rotation).
- Modality comparison view (run the same content through multiple modalities and diff the predictions).
- Persistent prediction history (currently in-memory only; clears on container restart).
- FastAPI endpoint alongside Gradio for programmatic access.
- Subject-level fine-tuning. Out of scope for v1; TRIBE supports it but only at training time.
- Subcortical voxels. TRIBE v2 is cortex-only.

## §13 Resolved decisions

| Decision | Value |
|---|---|
| Subdomain | `tribev2.ws.coursebite.ai` |
| Text input | Enabled by default (`ENABLE_TEXT=true`) |
| Modality count (inference) | 3 |
| Input tab count (UI) | 4 (image → video conversion) |
| GPU strategy | Single device `cuda:0` |
| Numpy pin | None (TRIBE requires 2.2.6) |
| HF CLI | `hf` (modern) |
| License | CC-BY-NC-4.0 |

## §14 References

- TRIBE v2 model card: https://huggingface.co/facebook/tribev2
- TRIBE v2 repository: https://github.com/facebookresearch/tribev2
- Paper: TODO (placeholder — fill once arXiv ID known)
- DataCamp tutorial: https://www.datacamp.com/tutorial/tribe-v2
- Meta FAIR blog post: TODO
- HCP-MMP1 (Figshare mirror): https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446
- Glasser et al. 2016 (Nature): https://www.nature.com/articles/nature18933
- Neurosynth: https://neurosynth.org
- NiMARE: https://nimare.readthedocs.io
- Karpathy CLAUDE.md guidelines (curated by Forrest Chang)
