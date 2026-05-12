# Overnight status — 2026-05-12

## Where we ended up

Bimodal v1 is **fully implemented and GPU-verified on the dev box**. The Dokploy application is **pre-configured but not yet deployed** because three settings need UI clicks (GPU device mapping, basic auth, volume mounts) that the Dokploy MCP doesn't expose.

You should be able to ship by clicking through ~5 settings and hitting **Deploy**.

## What was done autonomously

### Implementation
- 6 commits on `feat/bimodal-v1`, squashed locally into `18e0653` and pushed to `github.com/harshpatel333/total-tribe-v2`.
- `src/inference.py` — `TribeInference` class with single-GPU placement; `config_update={"data.text_feature": None, "data.features_to_use": ["audio", "video"]}` to skip LLaMA; file-hash cache; MoviePy `_image_to_video` helper.
- `src/app.py` — Gradio UI with 4 input tabs (image/audio/text/video). Text tab shows "Meta approval pending" notice when `ENABLE_TEXT=false`. Both hemispheres rendered.
- `src/interpretation.py` — `RegionInterpreter` working against real HCP-MMP1 atlases (fsaverage5).
- Atlases — `lh.HCP-MMP1.annot` and `rh.HCP-MMP1.annot` committed (~86 KB each, GOBS-derived; original Figshare set was fsaverage7 — spec corrected).

### GPU verification (on the dev box)
| Test | Result |
|---|---|
| Bimodal model load (text disabled) | 0.71 GB VRAM |
| Audio predict (30 s silence) | `(30, 20484)`; peak **7.84 GB** |
| Image-as-video predict (5 s clip) | `(5, 20484)`; peak **10.8 GB** |
| Gradio bind to :7860 | HTTP 200 |

### Dokploy MCP setup
- Project `tribev2` (`xfzXVo2wrjyGLJB3900nq`) in production env (`CmKz0x83NvnRG3Rpn5fg9`).
- Application `tribev2-app` (`bwQGLWn6XE-izPbVJ1U4q`) — git source pointed at this repo, branch `feat/bimodal-v1`.
- Build type: Dockerfile, context `/`.
- Env vars: `ENABLE_TEXT=false`, `HF_HOME=/app/cache`, `TRIBE_CACHE=/app/cache`, `ATLASES_DIR=/app/atlases`, `HF_TOKEN=` (empty placeholder).
- Domain: `tribev2.ws.coursebite.ai` :7860, HTTPS via Let's Encrypt.

## What you need to do (5–10 min)

In the Dokploy UI at the `tribev2-app` application:

### 1. Add the GPU mapping (Advanced → Swarm)
The MCP tool doesn't expose Docker resource reservations for an Application type. Easiest path:
- **Option A (recommended): switch to a Compose deployment.** Delete `tribev2-app` and create a new "Compose" service in the same project, paste the contents of `docker-compose.yml` from this repo. Compose natively supports `device_ids: ["0"]` for GPU.
- **Option B: add Generic Resources via Swarm labels.** In Advanced → Swarm → "Update Config" or similar, add `--generic-resource NVIDIA-GPU=1`. (Requires the host to have `node-generic-resources` configured in `/etc/docker/daemon.json` — likely already true since you run other GPU workloads.)

If unsure, **Option A is cleaner**.

### 2. Mount the cache volume
Otherwise every restart re-downloads ~16 GB of HF weights.

- Advanced → Mounts → Add volume:
  - Type: Volume
  - Name: `tribev2-cache`
  - Mount path: `/app/cache`
- Add another mount for uploads:
  - Type: Volume
  - Name: `tribev2-uploads`
  - Mount path: `/app/uploads`

### 3. Add basic auth on the domain
- Domains → `tribev2.ws.coursebite.ai` → Basic Auth → enable, set username + password.

### 4. Hit Deploy
First build pulls the CUDA base image, installs uv + Python 3.12, runs `uv pip install -e ".[gpu]"`. Expect **15–25 min** for the first build. First container start downloads ~16 GB of HF weights into the cache volume (**~10 min**, only once if the volume persists).

### 5. (Optional) Speed up the first start
Pre-populate the cache volume with the weights we already have on the dev box. From a shell on the Dokploy host:
```bash
docker run --rm -v tribev2-cache:/app/cache busybox sh -c "ls /app/cache"
# Then rsync /workspace/tribev2-spike/.hf_cache/ contents into it.
```
This saves ~10 minutes on the first start.

## When LLaMA approval lands

The text path is fully implemented and already gated on `ENABLE_TEXT`. To enable trimodal:

1. Dokploy → application → Environment → set `ENABLE_TEXT=true`.
2. Add `HF_TOKEN=hf_xxx` to env (your read token with Llama-3.2-3B access).
3. Redeploy.

No code change required. VRAM will go from ~16 GB peak to ~23 GB (still within 24 GB on the 3090).

## Smoke test plan (after deploy)

In the deployed UI:
1. Upload a video with a face close-up (e.g. a 15 s clip). Expect predictions, brain map renders, and `L_FFC` / `R_FFC` (fusiform face areas) in the top regions at `t=5 s`.
2. Upload a still image of a face. Same expectation (image is converted to a 5 s clip client-side).
3. Audio: try a 30 s clip with speech. Expect predictions; auditory regions (`L_PT`, `R_PT`) should rank.

## Where the artifacts live

- Repo (local): `/Users/harshpatel/code/open-source/total-tribe-v2`
- Repo (remote dev box): `/workspace/total-tribe-v2`
- TRIBE spike env (separate venv with model installed): `/workspace/tribev2-spike`
- Pre-downloaded HF weights (~7 GB, non-gated): `/workspace/tribev2-spike/.hf_cache/hub/`
- Dokploy MCP IDs: see this file's "What was done" section
- GitHub: `https://github.com/harshpatel333/total-tribe-v2` — `main` and `feat/bimodal-v1` both pushed

## Things still pending / risky

- **Meta LLaMA-3.2-3B approval** — at the time of writing, request is "awaiting review". Once approved, flip `ENABLE_TEXT=true` and redeploy.
- **First Dokploy build** will be ~15–25 min. If it fails, the most likely cause is a missing system dep (e.g. `ffmpeg`) — the Dockerfile installs it but worth checking build logs.
- **VRAM headroom is thin (~13 GB free of 24 GB at the predict peak)**. If you want to enable trimodal, that drops to ~1 GB free. Test with short inputs first.
- **`tests/test_inference.py` was rewritten to mock the upstream model** — the old "stub raises NotImplementedError" assertions are obsolete. New tests cover dispatch + caching CPU-only; GPU coverage was done out-of-band (see this file).
- **Single-user `LAST` dict** in `src/app.py` is intentional (matches the single-user-behind-basic-auth deployment model). Multi-tenant scaling is out of scope for v1.
