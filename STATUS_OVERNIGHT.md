# Overnight status — 2026-05-12

## Bottom line

**`https://tribev2.ws.coursebite.ai/` is LIVE and all three modalities verified end-to-end via Playwright on the deployed UI.** Image, video, and audio (speech) each produce neurally distinct, semantically appropriate brain maps. Text remains gated on Meta LLaMA-3.2-3B approval.

## Deploy

- Live URL: https://tribev2.ws.coursebite.ai/ (HTTPS via Let's Encrypt, Traefik)
- Final commit: `14178e6` on `feat/bimodal-v1` (deployment `YUK00OhN4OBqnrO6G6VbI`)
- Build sequence — three fixes landed during the night:
  1. `.dockerignore` was excluding `README.md` but `pyproject.toml` needs it for `readme = "README.md"` metadata. Fixed in `8a682c9`.
  2. Ubuntu 22.04 (jammy) doesn't ship `python3.12` in apt. Switched to `uv python install 3.12` matching the dev box. Fixed in `dc59910`.
  3. TRIBE's `ExtractWordsFromAudio` shells out to `uvx whisperx ...`; the subprocess couldn't find `uvx` on PATH, and Gradio swallowed the resulting `FileNotFoundError`, leaving the UI stuck on a phantom "processing" indicator. Fixed in `14178e6` (Dockerfile symlinks `uv`/`uvx` into `/usr/local/bin` + defensive PATH prepend in `src/inference.py`).
- Build time: ~5–7 min once caches warmed.

## End-to-end Playwright smoke tests

### Image (synthetic face PNG)
- VRAM: baseline → **14014 MiB** (V-JEPA2 + W2v-BERT + fusion)
- `T=5` segments (5 s @ 1 Hz)
- Top region: **R_MT** (-0.73, motion/vision/MT terms)
- Screenshot: `.playwright-mcp/tribev2-deployed-ui-image-predict.png`

### Video (8 s mp4, looped face, 4 fps)
- VRAM: baseline → **14006 MiB**
- `T=8` segments
- Top regions: **R_MT** (-0.59) + bilateral **R_VMV3** (+0.44) / **L_VMV3** (+0.39) — ventromedial visual area firing for faces
- Screenshot: `.playwright-mcp/tribev2-deployed-ui-video-predict.png`

### Audio (32 s synthesized speech via macOS `say`)
- VRAM stays low (~4.5 GB peak) — audio path uses W2v-BERT (~1 GB) without V-JEPA2
- **First-run cold start: ~12 min** because WhisperX downloads `large-v3` (~3 GB) + `WAV2VEC2_ASR_LARGE_LV60K_960H` (~1 GB) on first invocation. Subsequent runs should be ~60–90 s once cached.
- Top regions — bilateral auditory cortex, **all positive activations**:
  1. R_PBelt (+0.35), R_LBelt (+0.31), L_A5 (+0.30), R_A5 (+0.29)
  2. R_A4 (+0.27), L_A4 (+0.27), L_LBelt (+0.25), R_TA2 (+0.25)
- This is the canonical speech-listening network. Compare to image/video where motion + ventral visual areas dominated — the model produces modality-appropriate signatures.
- Screenshot: `.playwright-mcp/tribev2-deployed-ui-audio-speech-predict.png`

### Text
- Blocked on Meta LLaMA-3.2-3B approval (still pending for `harshpatel333` as of 05:33 UTC).
- When approved: in Dokploy → Environment, set `ENABLE_TEXT=true` and `HF_TOKEN=hf_...`, redeploy. No code change. VRAM at predict will jump from ~14 GB (image/video) to ~22 GB to add the LLaMA encoder — still well within the 24 GB ceiling.

## What's left for morning (low-effort UI clicks)

1. **(Recommended) Cache volume** — currently the container's `/app/cache` lives on overlayfs, which means every redeploy re-downloads ~16 GB of HF weights. Add a Dokploy volume mount: `tribev2-cache → /app/cache`. Same for `tribev2-uploads → /app/uploads`. Without this, the audio cold-start cost (~12 min for WhisperX) repeats every restart.
2. **(Recommended) Basic auth** — the domain is currently public. Dokploy → Domains → `tribev2.ws.coursebite.ai` → Basic Auth → enable, set user/pass.
3. **(When Meta approves)** — flip `ENABLE_TEXT=true` + add `HF_TOKEN`, redeploy. ~7 min.

## Dokploy handles

- Project `tribev2` (`xfzXVo2wrjyGLJB3900nq`) → Production env (`CmKz0x83NvnRG3Rpn5fg9`)
- Application `tribev2-app` (`bwQGLWn6XE-izPbVJ1U4q`) → service name `app-program-digital-pixel-3lwe9d`
- Domain `tribev2.ws.coursebite.ai` (`z5PA0DLiEYfuIjQfM81TC`)

## Git state

- `main` at `4270fec` (scaffold) — pushed
- `feat/bimodal-v1` at `14178e6` — pushed and deployed
- Tag `v0.1.0` not yet applied; that comes after the morning review per `CLAUDE.md` § 10.11.

## Memory artifacts saved

In `~/.claude/projects/-Users-harshpatel-code-open-source-total-tribe-v2/memory/`:
- `autonomous-overnight-plan.md` — full overnight protocol + heartbeat trace
- `v1-scope-decision.md` — trimodal-with-image-as-video locked
- `tribev2-modality-encoders.md` — model internals from the spike
- `remote-workspace-layout.md` — dev box paths
- `MEMORY.md` — index

## What surprised me

- TRIBE shells out to `uvx whisperx` rather than calling whisperx as a Python library. This is fragile (PATH dependency) and slow (first-run download in subprocess can't be cleanly progress-reported to Gradio). Worth filing upstream if you ever want to.
- TRIBE releases encoder VRAM aggressively between predict calls — image/video paths drop back to ~baseline after each predict. That means every new predict pays the encoder load cost (~2–3 s for V-JEPA2 + W2v-BERT). Not a blocker but noticeable.
- `image_feature` in the published TRIBE config is a phantom (claimed to use DINOv2 but never wired into the fusion). Confirmed by inspecting `model_build_args.feature_dims` which only has audio/text/video. Documented in `docs/SPIKE_FINDINGS.md`.
