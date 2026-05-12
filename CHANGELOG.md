# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-12

First public release. Bimodal v1 is deployed and verified end-to-end (image / audio / video). Text is plumbed but inert until Meta approves Llama-3.2-3B access for the maintainer's HuggingFace account — flipping `ENABLE_TEXT=true` enables it with no code change.

### Added

- Repo skeleton: tech spec (`docs/TECH_SPEC.md`), 5 ADRs (`docs/DECISIONS/`), CONTRIBUTING/SECURITY/CODE_OF_CONDUCT, Karpathy-style behavioral guidelines in `CLAUDE.md` (attribution: Forrest Chang).
- **Bimodal v1 inference engine** (`src/inference.py`). `TribeInference` loads the real `facebook/tribev2` checkpoint with `config_update={"data.text_feature": None, "data.features_to_use": ["audio", "video"]}` when `ENABLE_TEXT=false`, so Llama-3.2-3B is never instantiated. Flipping `ENABLE_TEXT=true` enables the full trimodal model with no code change.
- **Gradio app** (`src/app.py`). Wires `TribeInference.predict` → `RegionInterpreter` → both-hemisphere `nilearn.plotting.view_surf` render + top-8 region table + timestep slider seeded at the BOLD peak (~+5 s).
- **Image input via client-side video conversion** (`_image_to_video`). Still images (jpg / jpeg / png / webp) become a 5 s silent MP4 via MoviePy 2.x and dispatch through the video pipeline.
- **Module-level file-hash predict cache** (SHA-256 keyed). Avoids re-running the upstream model for repeat inputs without pinning the `TribeInference` instance.
- **HCP-MMP1 atlases for fsaverage5** (`atlases/lh.HCP-MMP1.annot`, `atlases/rh.HCP-MMP1.annot`, ~86 KB each) and the `scripts/fetch_atlases.py` that produced them from `poldrack/GOBS` GIFTI labels.
- **Region interpreter** (`src/interpretation.py`) with a hand-curated `atlases/parcel_lookup.json` mapping the 50 most-likely-active HCP-MMP1 parcels to Neurosynth-derived semantic terms.
- **Dokploy-ready container** — `Dockerfile` installs Python 3.12 via uv (jammy ships 3.10), symlinks `uv`/`uvx` into `/usr/local/bin` for subprocess discoverability, and `docker-compose.yml` requests `device_ids: ["0"]` for GPU passthrough.
- **Smoke test screenshots** under `docs/images/` from the first-deploy verification: image predict, video predict, audio (speech) predict.

### Changed

- Install path: `tribev2` from `git+https://github.com/facebookresearch/tribev2.git` (not PyPI; `[gpu]` optional extra).
- No `numpy<2.1.0` pin; TRIBE 0.1.0 requires `numpy==2.2.6`.
- Single-GPU placement (`cuda:0`) replaces the original 3-GPU per-encoder strategy. Spike confirmed the upstream demo API exposes no per-encoder attributes (see `docs/SPIKE_FINDINGS.md` § 2).
- HF auth uses `hf` (not deprecated `huggingface-cli`).
- Brain renderer outputs both hemispheres.
- `pyproject.toml` sets `tool.hatch.metadata.allow-direct-references = true` to permit the git-based `tribev2` reference.
- `RegionInterpreter` normalises GOBS-style annot labels (`L_V1_ROI`) to match the bare keys in `parcel_lookup.json` (`V1`).

### Fixed

- `.dockerignore` was excluding `README.md`, breaking `uv pip install -e .` because `pyproject.toml` declares `readme = "README.md"`. README is now allowed through.
- TRIBE's `ExtractWordsFromAudio` shells out to `uvx whisperx ...` via `subprocess.run`. When `uvx` wasn't on the inherited PATH, the subprocess errored with `FileNotFoundError` and Gradio swallowed it, leaving the UI stuck on a phantom "processing" indicator. Fixed by symlinking `uv`/`uvx` into `/usr/local/bin` in the Dockerfile **and** prepending the candidate uv install dirs to `os.environ["PATH"]` at `src/inference.py` import time.

### Verified at first deploy

End-to-end through the deployed UI, behind Traefik + Let's Encrypt:

| Modality | VRAM peak | T (segments) | Top region(s) | Sanity |
| --- | --- | --- | --- | --- |
| Image (synthetic face) | 14014 MiB | 5 | `R_MT` (-0.73, motion/vision) | sensible for static loop |
| Video (8 s face loop) | 14006 MiB | 8 | `R_MT` (-0.59) + bilateral `R_VMV3`/`L_VMV3` (+0.44/+0.39) | ventral visual area firing for faces |
| Audio (32 s speech) | ~4.5 GB | 33 | All 8 in bilateral auditory cortex: `R_PBelt`, `R_LBelt`, `L/R_A5`, `L/R_A4`, `L_LBelt`, `R_TA2` (all +0.25 to +0.35) | canonical speech-listening network |

See `docs/RISKS.md` for closed/open risks.
