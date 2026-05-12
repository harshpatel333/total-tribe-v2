# ADR-0002: Single-GPU placement (`cuda:0`) for inference

- **Status:** Accepted
- **Date:** 2026-05-11

## Context

The original tech spec proposed a three-GPU layout, placing each encoder (text / audio / video) on a separate device to spread the ~28–32 GB of model weights across three RTX 3090s. The spike (see `docs/SPIKE_FINDINGS.md` §2) showed this layout is impossible against the public TRIBE v2 API:

- `TribeModel.from_pretrained` accepts a **single** `device` argument (`"cuda"`, `"cuda:0"`, `"cpu"`, or `"auto"`). There is no per-encoder device argument.
- `TribeModel` exposes **no** per-encoder attributes. There is no `model.text_encoder`, `model.audio_encoder`, `model.video_encoder`, or `model.fusion_transformer`. The actual network is built monolithically inside `model._model` from `xp.brain_model_config.build(**build_args)`.
- Upstream FSDP support exists in the codebase but is gated to training only; the inference path is single-device.

The DataCamp tutorial's "40 GB minimum" was a single-card recommendation including activation headroom for longer clips. Spike VRAM accounting on the actual checkpoint layout reduces this to ~23 GB peak on one device under a 30 s input cap.

## Decision

Inference runs on a single device, hard-coded to `"cuda:0"`. The wrapper enforces a **≤30 s input duration cap** to stay under the 24 GB ceiling at peak (LLaMA 7 GB + V-JEPA2 14 GB + W2v-BERT 1 GB + fusion 1 GB + working activations ≈ 23 GB). The cap is enforced at the inference layer (`TribeInference.predict`) before any model call, with a clear exception message.

Multi-GPU configurations on the host are out of scope; any other GPU workloads on the box (e.g. Ollama) must be paused or pinned to other devices manually before serving (see `docs/SPIKE_FINDINGS.md` §10).

## Consequences

**Positive:**

- Matches the upstream API exactly. No introspection, no monkey-patching, no fragility against upstream releases.
- Predictable VRAM accounting: total target is one number, not three.
- Hardware floor is one 24 GB consumer card. Same card that runs typical local LLMs already.

**Negative:**

- Strict input duration cap. Long-form stimuli (a feature film, a lecture) cannot be processed in one pass. v1 explicitly does not target this.
- No throughput scaling from extra GPUs. A second card on the box is idle from this wrapper's perspective. Since the deployment is single-user, this is acceptable.

## Alternatives considered

- **HF Accelerate `dispatch_model` with a custom `device_map`.** Rejected — would require name-based introspection of the monolithic `_model.brain_model` to identify which submodule corresponds to which encoder, then a hand-crafted `device_map`. Brittle against upstream releases, throughput-penalized for batch-size-1 interactive use (cross-device tensor transfers dominate), and offers no VRAM benefit on a 24 GB card we already fit on.
- **A single 40 GB+ card** (A100 40 GB, A6000 48 GB, H100 80 GB). Rejected — cost. The 24 GB target works.
- **CPU fallback.** Rejected — too slow for interactive use. A 30 s video prediction would take minutes on CPU and would not surface meaningful interactive behavior. We keep `device="cpu"` as a debug option but it is not a supported path.
