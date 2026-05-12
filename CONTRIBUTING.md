# Contributing to total-tribe-v2

Thank you for your interest. This is a small research-grade open-source project and pull requests are welcome. Please read this document and [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) before opening one.

## Dev setup (one command)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```

You will need [`uv`](https://github.com/astral-sh/uv) installed. If you do not have a GPU, you can still develop and run non-GPU tests; the marker system (`-m "not gpu"`) skips anything that requires CUDA.

## Tests

Run the non-GPU test suite:

```bash
pytest tests/ -v -m "not gpu"
```

GPU-gated tests live behind `@pytest.mark.gpu` and run in CI only when a GPU runner is configured. Treat them as integration tests, not unit tests.

## Lint and format

Check (no changes written):

```bash
ruff check . && black --check . && isort --check-only .
```

Auto-format:

```bash
ruff check --fix . && black . && isort .
```

The repository targets Python 3.12, line length 100, and the project's `pyproject.toml` is the source of truth for tool configuration.

## Pull request checklist

This mirrors `.github/PULL_REQUEST_TEMPLATE.md`. Before requesting review, confirm:

- [ ] Tests added or updated for the change (and they pass locally).
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` with a short, specific entry.
- [ ] An ADR added under `docs/DECISIONS/NNNN-<title>.md` if the change is architectural (new dependency, new top-level abstraction, new public API surface).
- [ ] Docs updated (`README.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, etc.) if behavior visible to users or operators changed.
- [ ] CI is green.

## Commits

We use [Conventional Commits](https://www.conventionalcommits.org/). Examples:

- `feat(ui): render both hemispheres in brain view`
- `fix(inference): release VRAM after each predict call`
- `docs(deploy): add Dokploy MCP playbook`
- `chore(deps): bump torch to 2.6.1`

Commit messages explain the *why*, not just the *what*.

## Review cadence

This is a side project. Reviews may take days, especially around weekends. Please do not ping for status before a week has passed; if it has been longer, a polite nudge is welcome.

## Code of Conduct

By participating, you agree to abide by [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).
