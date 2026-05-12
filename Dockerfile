FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache \
    TRIBE_CACHE=/app/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv for fast installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY atlases ./atlases

# Install with [gpu] extra (pulls tribev2 from git + torch 2.6.0)
RUN uv pip install --system -e ".[gpu]"

# Atlas fetch happens at first run (network at build is unreliable behind some firewalls)
# scripts/fetch_atlases.py is idempotent

EXPOSE 7860

CMD ["python", "-m", "src.app"]
