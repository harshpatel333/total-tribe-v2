FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache \
    TRIBE_CACHE=/app/cache

# Stock jammy ships python 3.10 only — we use uv to install 3.12 below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv, then a managed Python 3.12, then a venv at /opt/venv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}"
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY atlases ./atlases

# Install with [gpu] extra (pulls tribev2 from git + torch 2.6.0)
RUN uv pip install -e ".[gpu]"

# Atlas fetch happens at first run if needed; scripts/fetch_atlases.py is idempotent
# (atlases/lh.HCP-MMP1.annot and rh.HCP-MMP1.annot are committed in the image)

EXPOSE 7860

CMD ["python", "-m", "src.app"]
