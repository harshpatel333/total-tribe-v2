# Overnight status — 2026-05-12

## Bottom line

**`https://tribev2.ws.coursebite.ai/` is LIVE.** Image + video pipelines verified end-to-end through the deployed UI with Playwright + nvidia-smi watching. Audio path needs one round of debug (WhisperX hangs on speech input — not a deploy problem). Text path is gated on Meta LLaMA-3.2-3B approval (env flip when it lands).

## What's verified

### Deploy
- Build commit: `dc59910` on `feat/bimodal-v1`. First build failed on `.dockerignore` excluding `README.md` (fixed in `8a682c9`) and apt's missing `python3.12` in jammy (fixed by switching to `uv python install 3.12` in `dc59910`).
- Container live behind Traefik + Let's Encrypt. `curl https://tribev2.ws.coursebite.ai/` → 200, `server: uvicorn`.
- GPU access works by default on this Dokploy host — no manual device mapping needed. Container sees all three RTX 3090s. Confirmed via nvidia-smi inside container.

### Image path
- Uploaded a synthetic 256×256 face PNG.
- Click → GPU 0 went **4288 → 14014 MiB** (~9.7 GB delta — V-JEPA2 + W2v-BERT + fusion).
- Returned `T=5` segments (5 s @ 1 Hz), both hemispheres rendered via Plotly iframes, region table populated with 8 entries.
- Top region: **R_MT** (-0.73) with terms `[motion, vision, MT]` — sensible for a static-face-as-loop stimulus.
- VRAM dropped back to 4288 after predict (TRIBE releases encoder VRAM between calls).
- Screenshot: `.playwright-mcp/tribev2-deployed-ui-image-predict.png`.

### Video path
- Uploaded an 8 s, 4 fps mp4 (looped face).
- GPU 0 went **4288 → 14006 MiB**.
- `T=8` segments. Different region table from image: top region still **R_MT** (-0.59), but `R_VMV3` (+0.44) and `L_VMV3` (+0.39) — bilateral ventromedial visual area — fire **positively**, which is what you'd expect for a face stimulus.
- Screenshot: `.playwright-mcp/tribev2-deployed-ui-video-predict.png`.

### Audio path
- **Silence (16 s of zeros) → no-op.** WhisperX correctly produced no Word events; data pipeline filtered out the empty input. UI returned without errors. Expected behavior; not a bug.
- **Speech (32 s of synthesized speech via `say`) → hung.** Gradio progress climbed past 300 s with the estimate stuck at 66 s. GPU 0 stayed at the 4288 MiB baseline the whole time — WhisperX never loaded onto the GPU. Likely WhisperX falling back to CPU and exceeding Gradio's worker timeout, or an exception inside the audio extractor not bubbling to the UI. Captured task #13 to follow up. Screenshot: `.playwright-mcp/tribev2-deployed-ui-audio-stuck.png`.

### Text path
- Blocked on Meta LLaMA-3.2-3B approval (still pending for `harshpatel333`).
- When approved: set `ENABLE_TEXT=true` and `HF_TOKEN=hf_...` in Dokploy env, redeploy. No code change needed.

## What still needs doing

1. **(Audio debug, task #13)** — check Dokploy container logs while a speech audio is processing. If WhisperX-on-CPU is the cause, force it to GPU via env var (`whisperx --device cuda`) or pin a smaller WhisperX model. The data pipeline is fine; this is purely an audio-extractor performance issue.
2. **(Volume mount, ~30 s in Dokploy UI)** — without `tribev2-cache → /app/cache`, every container restart re-downloads ~16 GB of HF weights into the container's overlayfs (slow startup, no persistence). Add the volume mount in Dokploy → Mounts.
3. **(Basic auth, ~30 s in Dokploy UI)** — domain is currently public. Add basic auth on `tribev2.ws.coursebite.ai` via Dokploy → Domains → Basic Auth.
4. **(LLaMA approval)** — when Meta grants access, flip `ENABLE_TEXT=true` + add `HF_TOKEN`, redeploy. VRAM at predict will go from ~14 GB to ~22 GB — still within 24 GB.

## Dokploy app handles

- Project: `tribev2` (`xfzXVo2wrjyGLJB3900nq`)
- App: `tribev2-app` (`bwQGLWn6XE-izPbVJ1U4q`)
- Production env: `CmKz0x83NvnRG3Rpn5fg9`
- Container appName: `app-program-digital-pixel-3lwe9d`

## Git state

- `main` at `4270fec` (scaffold) — pushed to origin
- `feat/bimodal-v1` at `dc59910` (bimodal impl + dockerignore fix + python-via-uv fix) — pushed to origin, deployed
- No PR opened — leaving that for you to review at your pace

## Memory artifacts

- `~/.claude/projects/-Users-harshpatel-code-open-source-total-tribe-v2/memory/`
  - `autonomous-overnight-plan.md` — overnight protocol + heartbeat plan
  - `v1-scope-decision.md` — trimodal v1 locked
  - `tribev2-modality-encoders.md` — spike findings about model internals
  - `remote-workspace-layout.md` — dev box layout
  - `MEMORY.md` — index
