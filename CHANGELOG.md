# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial repo skeleton.
- Revised tech spec (`docs/TECH_SPEC.md` v2).
- Spike-derived corrections vs original spec (see `docs/SPIKE_FINDINGS.md`).
- Five ADRs under `docs/DECISIONS/`.
- Karpathy-style behavioral guidelines in `CLAUDE.md` (attribution: Forrest Chang).
- Gradio UI stubs for 4 input modalities (image / audio / text / video).
- HCP-MMP1 region lookup scaffolding (`atlases/parcel_lookup.json` + build script).
- `Dockerfile` and `docker-compose.yml` targeted at Dokploy deployment.
- **Bimodal v1 implementation**: `TribeInference` loads the real `facebook/tribev2`
  checkpoint with `config_update={"data.text_feature": None, "data.features_to_use":
  ["audio", "video"]}` when `ENABLE_TEXT=false`, so LLaMA-3.2-3B is never
  instantiated. Flipping `ENABLE_TEXT=true` enables the full trimodal model with
  no code change.
- End-to-end Gradio app wires `TribeInference.predict` → `RegionInterpreter` →
  both-hemisphere `nilearn.plotting.view_surf` render + top-8 region table +
  timestep slider seeded at the BOLD peak (~+5 s).
- `_image_to_video` converts still images (jpg / jpeg / png / webp) to a 5 s
  silent MP4 via MoviePy 2.x and dispatches through the video pipeline.
- Module-level file-hash predict cache (SHA-256 keyed) avoids re-running the
  upstream model for repeat inputs without pinning the `TribeInference` instance.
- `scripts/fetch_atlases.py` now downloads fsaverage5 HCP-MMP1 GIFTI labels from
  the `poldrack/GOBS` repo and converts in-place to FreeSurfer `.annot`; the
  previous Figshare URLs pointed at fsaverage7 and would not match our pipeline.
- HCP-MMP1 `.annot` files for both hemispheres committed under `atlases/` (~86 KB
  each, CC-BY-NC compatible via WU-Minn HCP Open Access terms; matches the
  project's CC-BY-NC-4.0 license).

### Changed

- Corrected install path: `tribev2` installed via `git+https://github.com/facebookresearch/tribev2.git` (not PyPI).
- Removed `numpy<2.1.0` pin; TRIBE 0.1.0 requires `numpy==2.2.6`.
- Single-GPU placement (`cuda:0`) replaces the original 3-GPU per-encoder strategy.
- HF auth uses `hf` CLI (not deprecated `huggingface-cli`).
- Brain renderer outputs BOTH hemispheres (was LH-only).
- Image input is converted to a short video clip client-side; routed through video pipeline.
- `pyproject.toml` now sets `tool.hatch.metadata.allow-direct-references = true`
  to permit the `tribev2 @ git+https://...` direct reference under `[gpu]`.
- `RegionInterpreter` normalises GOBS-style annot labels (`L_V1_ROI`) so they
  match the bare keys in `parcel_lookup.json` (`V1`).

### Changed

- Corrected install path: `tribev2` installed via `git+https://github.com/facebookresearch/tribev2.git` (not PyPI).
- Removed `numpy<2.1.0` pin; TRIBE 0.1.0 requires `numpy==2.2.6`.
- Single-GPU placement (`cuda:0`) replaces the original 3-GPU per-encoder strategy.
- HF auth uses `hf` CLI (not deprecated `huggingface-cli`).
- Brain renderer outputs BOTH hemispheres (was LH-only).
- Image input is converted to a short video clip client-side; routed through video pipeline.
