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

### Changed

- Corrected install path: `tribev2` installed via `git+https://github.com/facebookresearch/tribev2.git` (not PyPI).
- Removed `numpy<2.1.0` pin; TRIBE 0.1.0 requires `numpy==2.2.6`.
- Single-GPU placement (`cuda:0`) replaces the original 3-GPU per-encoder strategy.
- HF auth uses `hf` CLI (not deprecated `huggingface-cli`).
- Brain renderer outputs BOTH hemispheres (was LH-only).
- Image input is converted to a short video clip client-side; routed through video pipeline.
