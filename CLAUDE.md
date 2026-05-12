# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## Overview

`total-tribe-v2` is a single-user, self-hosted Gradio UI that wraps Meta FAIR's TRIBE v2 brain-encoding model. The wrapper takes text, audio, or video stimuli (plus an image-as-video-clip path) and returns predicted cortical activations on the `fsaverage5` surface, rendered in a browser with an interpretable region table. The system targets a single 24 GB GPU and enforces a ≤30 s input cap to stay under that ceiling.

## Repo layout

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the component breakdown and [`docs/TECH_SPEC.md`](./docs/TECH_SPEC.md) for the implementation-level spec. ADRs live under [`docs/DECISIONS/`](./docs/DECISIONS/).

## Tech stack

- **Language:** Python 3.12 (`uv` for environment management)
- **ML:** PyTorch 2.6 (CUDA 12.4 build), `tribev2` (installed from git), `transformers`, `huggingface_hub`
- **UI:** Gradio
- **Neuroimaging:** `nilearn`, `nibabel`
- **Media:** MoviePy (for image-to-video client-side conversion)
- **Container:** Docker with `nvidia-container-toolkit`; deployed via Dokploy

## Run locally

After populating `.env` (see `README.md` § Quickstart):

```bash
docker compose up -d
```

## Run tests

```bash
pytest tests/ -v -m "not gpu"
```

GPU-gated tests use `@pytest.mark.gpu` and are skipped by default.

## Coding conventions

- **Formatter:** Black with default config (line length 100, target `py312`).
- **Linter:** Ruff with the project's `pyproject.toml` rules.
- **Imports:** isort with Black profile.
- **Types:** Type hints required on all public functions (anything not prefixed `_`).
- **Logging:** Use `logging.getLogger(__name__)`. **No `print()`** in library code.
- **Exceptions:** No bare `except:`. No broad `except Exception:` without re-raising or adding context.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) are required. The commit message explains the *why*, not just the *what*.

## When making changes — checklist

- [ ] Update `CHANGELOG.md` under `## [Unreleased]`.
- [ ] If the change is architectural (new dependency, new abstraction, new public API surface), write an ADR under `docs/DECISIONS/NNNN-<slug>.md`.
- [ ] Add or update tests. Mark GPU-only tests with `@pytest.mark.gpu`.
- [ ] Update docs if behavior visible to users or operators changed.

## When uncertain

Write an ADR proposing 2–3 options first. Implement only after the option is confirmed. ADRs are cheap; reversed architectural decisions are expensive.

# Behavioral guidelines
_Adapted from Karpathy-style operating principles; curated for this repo by Forrest Chang._

You are a careful, senior engineer working on a small research-grade open-source project. The expectations:

- **Read before writing.** Before changing code, read the surrounding module and any directly-coupled tests. Before changing docs, read the doc and its referenced ADRs.
- **Prefer surgical edits over rewrites.** If a function needs a 3-line change, do not rewrite the file. If a doc needs a paragraph, do not restructure the doc.
- **Don't expand scope.** If the task is "fix the bug in X", do not also refactor Y. If the user asks for a one-off, do not generalize it.
- **Ask before architectural choices.** New abstractions, new dependencies, new top-level files, new public API surface — propose first, then implement after confirmation. Write an ADR for any decision that future-you would have to re-derive from code.
- **Make failure modes loud.** Use real exceptions with clear messages, not `pass` or silent fallbacks. Log at appropriate levels (`logging.getLogger(__name__)`).
- **Cite what changed and why.** Every commit message follows Conventional Commits and explains the *why*, not just the *what*. Every PR updates `CHANGELOG.md` under `[Unreleased]`.
- **Tests are part of the change.** No "I'll add tests later." If a function is non-trivial, it ships with at least one test. Mark GPU-only tests with `@pytest.mark.gpu`.
- **The model is gated.** Do not call `TribeModel.from_pretrained` in CI, in tests, or in any code path that runs without `HF_TOKEN` set and LLaMA-3.2-3B access granted. Surface the gate as a clear `NotImplementedError` until approval lands.
- **Single user, single GPU.** Module-level state (`LAST`) is intentional for the slider UX. Do not generalize to multi-tenant without an ADR.
- **Respect the license.** This project is CC-BY-NC-4.0 because upstream TRIBE v2 is. Do not add code paths that imply or facilitate commercial use.
- **Be specific in prose.** "Fixed a bug" is not a commit message. "Render both hemispheres in brain view (was LH-only)" is.

**These guidelines are working if:** a future maintainer can clone the repo, read this file, and within an hour know how to make a safe change.
