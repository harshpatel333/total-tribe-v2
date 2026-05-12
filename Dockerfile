FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache \
    TRIBE_CACHE=/app/cache \
    TORCH_HOME=/app/cache/torch \
    PRELOAD_DIR=/opt/preloaded

# Stock jammy ships python 3.10 only — we use uv to install 3.12 below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv, then a managed Python 3.12, then a venv at /opt/venv.
# Symlink uv/uvx into /usr/local/bin so they are on PATH regardless of how
# the container is invoked (Dokploy/Swarm sometimes scrubs ENV PATH for the
# entrypoint, and TRIBE's audio extractor shells out to `uvx whisperx ...`).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -sf /root/.local/bin/uv  /usr/local/bin/uv \
 && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx
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

# ---------------------------------------------------------------------------
# Pre-cache model weights to /opt/preloaded (baked into the image).
# Entrypoint copies these into /app/cache on first start, so:
#   - no volume mount    → /app/cache has weights from this image
#   - mounted named vol  → first start seeds the volume; subsequent starts
#                          find /app/cache populated and skip the copy
#
# Without this, every Dokploy redeploy redownloads ~10 GB on the first
# predict (~10-12 min cold start, observed during overnight verification).
# Cost: +~11 GB image, +~10 min build. Worth it for predictable startup.
# ---------------------------------------------------------------------------

# HuggingFace models. ENABLE_TEXT=false at v0.2.0, so we skip the gated
# meta-llama/Llama-3.2-3B; when that's approved, baking it lands in a
# follow-up commit.
RUN HF_HOME="$PRELOAD_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in [
    "facebook/tribev2",
    "facebook/vjepa2-vitg-fpc64-256",
    "facebook/w2v-bert-2.0",
    "facebook/dinov2-large",
]:
    print(f"==> {repo}")
    snapshot_download(repo, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
PY

# WhisperX large-v3 (~3 GB faster-whisper) + WAV2VEC2_ASR_LARGE_LV60K_960H
# alignment model (~1 GB torchaudio). Triggering downloads via a real
# `uvx whisperx ...` call on a 1 s silent WAV is the most reliable way —
# faster-whisper's lazy loader and torchaudio's pipeline both fire here.
RUN python - <<'PY'
import wave
with wave.open("/tmp/sil.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000)
PY

RUN HF_HOME="$PRELOAD_DIR" TORCH_HOME="$PRELOAD_DIR/torch" \
    uvx --no-cache whisperx /tmp/sil.wav \
        --model large-v3 --device cpu --compute_type int8 --batch_size 1 \
        --output_dir /tmp/wx-warmup --output_format json \
        --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --language en \
    2>&1 | tail -5 \
 || echo "warmup whisperx call returned non-zero — that's fine if models still downloaded"
RUN rm -rf /tmp/sil.wav /tmp/wx-warmup

EXPOSE 7860

# Entrypoint seeds /app/cache from /opt/preloaded when /app/cache is empty
# (covers fresh volume mounts and image-only runs alike).
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["python", "-m", "src.app"]
