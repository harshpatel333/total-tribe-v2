# Architecture

## Overview

`total-tribe-v2` is a thin service in front of Meta's TRIBE v2 brain-encoding model. The browser sends a stimulus (text, audio, video, or an image which is client-side-converted to a short video clip) to a Gradio server. The server invokes `TribeModel.predict()` once per request, holding the resulting `(T, 20484)` prediction array in a module-level cache so a UI time-slider can re-render different timepoints without re-running inference. Two interpretation layers run on top of the raw predictions: an `nilearn` surface renderer (both hemispheres) and a `RegionInterpreter` that aggregates per-vertex activations into HCP-MMP1 parcels and looks up Neurosynth cognitive terms.

## Request sequence

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant T as Traefik (Dokploy)
    participant G as Gradio (src.app)
    participant I as TribeInference
    participant M as TribeModel
    participant R as RegionInterpreter
    participant V as Surface renderer (nilearn)

    B->>T: HTTPS POST /api/predict (stimulus, basic-auth)
    T->>G: forward request
    G->>I: predict(modality, path)
    I->>M: get_events_dataframe(...)
    M-->>I: events DataFrame
    I->>M: predict(events, verbose=False)
    M-->>I: (preds: np.ndarray[T,20484], segments)
    I-->>G: cache LAST = preds; return preds
    G->>R: top_regions(preds[t])
    R-->>G: top-K parcel table (+ Neurosynth terms)
    G->>V: view_surf(LH) + view_surf(RH)
    V-->>G: iframe HTML (both hemispheres)
    G-->>T: HTML + JSON table
    T-->>B: response (iframe + table)
```

## Components

The implementation is partitioned into three subsystems that mirror the sections in [`docs/TECH_SPEC.md`](./TECH_SPEC.md):

- **Inference (§6.1).** `src/tribe_inference.py` owns the single `TribeModel` instance, the `(file_hash, modality) → preds` cache, and the ≤30 s input-duration gate. The model is constructed once at startup via `TribeModel.from_pretrained("facebook/tribev2", cache_folder=..., device="cuda:0")`. Per [ADR-0002](./DECISIONS/0002-single-gpu-placement.md), `cuda:0` is the only device the wrapper is allowed to touch.

- **Interpretation (§6.2).** `src/region_interpreter.py` loads the HCP-MMP1 `.annot` files (resampled to `fsaverage5`), maps each cortical vertex to a parcel, aggregates per-vertex activation magnitudes per parcel per timepoint, and joins the result against `atlases/parcel_lookup.json` to surface Neurosynth-derived cognitive terms. See [`docs/INTERPRETATION.md`](./INTERPRETATION.md) for the data flow and the lookup-table schema.

- **UI (§6.3).** `src/app.py` builds the Gradio interface (four input tabs, a time slider, a `gr.HTML` brain pane, a `gr.Dataframe` region table). The image tab routes through MoviePy → short video clip → video pipeline (see [ADR-0005](./DECISIONS/0005-image-as-video-clip-conversion.md)).

## Volumes

Two persistent volumes are mounted into the container:

- **`/app/cache`** — Hugging Face model cache plus the TRIBE v2 weights cache (`cache_folder` argument to `from_pretrained`). First request after a fresh deploy populates this; subsequent runs are warm.
- **`/app/uploads`** — Working directory for inbound media. The server writes user-supplied audio/video here before passing the path to `get_events_dataframe(...)`. Eviction is best-effort; in v1 it is cleared at container restart.

## File layout

```text
total-tribe-v2/
├── README.md                  # entry point; quickstart and links
├── LICENSE                    # CC-BY-NC-4.0 full text
├── CHANGELOG.md               # Keep-a-Changelog format
├── CONTRIBUTING.md            # dev setup, lint, PR checklist
├── CODE_OF_CONDUCT.md         # Contributor Covenant 2.1
├── SECURITY.md                # private security advisory channel
├── CLAUDE.md                  # agent guidance + behavioral guidelines
├── Dockerfile                 # CUDA 12.4 base, Python 3.12, uv install
├── docker-compose.yml         # single service, GPU passthrough, no Traefik labels
├── pyproject.toml             # deps, ruff/black/isort/pytest config
├── .env.example               # HF_TOKEN, ENABLE_TEXT, BASIC_AUTH_*
├── src/                       # application code (inference / interpretation / UI)
├── scripts/                   # smoke_test.py, build_neurosynth_lookup.py, etc.
├── atlases/                   # HCP-MMP1 .annot files + parcel_lookup.json
├── tests/                     # pytest; @pytest.mark.gpu for GPU-gated tests
└── docs/
    ├── ARCHITECTURE.md        # (this file)
    ├── DEPLOYMENT.md          # Dokploy walkthrough + MCP playbook
    ├── INTERPRETATION.md      # region lookup pipeline + JSON schema
    ├── RISKS.md               # open verification gates
    ├── TECH_SPEC.md           # v2 implementation spec (owned elsewhere)
    ├── SPIKE_FINDINGS.md      # corrections vs original spec (owned elsewhere)
    ├── images/                # screenshots (post-deploy)
    └── DECISIONS/             # ADRs (0001..0005 at scaffold time)
```
