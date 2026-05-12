# ADR-0001: No Ollama backend; use upstream `TribeModel` directly

- **Status:** Accepted
- **Date:** 2026-05-11

## Context

An early framing of this project asked whether TRIBE v2 could be served behind an Ollama-style HTTP backend, so that the UI could talk to it via the OpenAI-compatible chat completions API and reuse the broader Ollama tooling ecosystem.

TRIBE v2 is **not** a language model. Its public inference API returns:

```python
preds, segments = model.predict(events=df, verbose=False)
# preds: np.ndarray, shape (T, 20484) on fsaverage5 at 1 Hz
```

That is a dense `float32` array of predicted cortical BOLD activations per vertex per second — it is not a token stream, has no vocabulary, and has no autoregressive structure. Ollama is a token-streaming LLM runtime; everything it exposes (chat completions, embeddings, streaming SSE deltas, tool calls) is shaped around discrete language tokens.

The two architectures are incompatible at the protocol level, and any bridge would be lossy and misleading.

## Decision

Use upstream `TribeModel.from_pretrained("facebook/tribev2", ...)` directly in-process. The Gradio server holds a single instance, calls `.predict()` synchronously per request, and returns the resulting `(T, 20484)` numpy array to the renderer. No Ollama process, no Ollama-compatible HTTP layer, no token streaming.

## Consequences

**Positive:**

- The wrapper is one process, one device, and one synchronous code path. Easy to reason about; matches upstream's own example usage.
- Output stays as numpy through the entire interpretation pipeline (no lossy conversion to and from text).
- We can use the exact upstream call signature and benefit directly from upstream version bumps.

**Negative:**

- No reuse of the existing Ollama tooling ecosystem. We do not get the model-management UI, REST surface, or quantization controls "for free." This is acceptable because none of them apply to a brain-encoding model anyway.
- A future "talk to your brain map" feature would still need a separate LLM. That LLM could run on Ollama on a different device; this ADR does not preclude that. It precludes only fronting TRIBE v2 itself through Ollama.

## Alternatives considered

- **Thin "LLM-ish" wrapper translating activations to text.** Rejected — misleading and lossy. Reducing a `(T, 20484)` activation map to a paragraph of prose hides the actual model output and conflates inference with interpretation. The region table (HCP-MMP1 + Neurosynth) already does the bounded, principled version of this.
- **Repackaging TRIBE v2 as a custom Ollama backend.** Rejected — out of scope for v1, requires writing a new Ollama runner type, and the underlying architecture mismatch means the Ollama protocol surface (token deltas, sampling temperature, etc.) has no meaning for the data we are producing.
