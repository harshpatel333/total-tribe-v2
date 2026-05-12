# Risks

Open verification gates from `docs/SPIKE_FINDINGS.md` (§ "What still needs verifying"), plus the LLaMA approval blocker. This is a living register; close rows by linking to the verifying commit/PR/ADR.

| #  | Risk                                                                                                       | Impact | Verification gate (closes when…)                                                                                                                                              | Owner    |
| -- | ---------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1  | **LLaMA-3.2-3B approval pending for `harshpatel333`**. Blocks all real-load testing.                       | HIGH   | Meta approves access on <https://huggingface.co/meta-llama/Llama-3.2-3B>; `hf auth whoami` shows the account; a smoke `TribeModel.from_pretrained` call succeeds with weights. | Harsh    |
| 2  | **VRAM peak under real load** — accounting predicts ~23 GB on a 24 GB card; needs measurement.             | HIGH   | A 30 s video prediction completes on the deploy host and `nvidia-smi`'s peak `memory.used` stays ≤24 GB across LH+RH render.                                                  | Harsh    |
| 3  | **`get_events_dataframe` actual column set**. Docstring claims `type`, `filepath`, `start`, `duration`, `timeline`, `subject`; we have not run it. | LOW    | A one-line invocation of `model.get_events_dataframe(video_path=...)` is captured in a test and the column list is asserted.                                                  | Claude   |
| 4  | **`CreateVideosFromImages` quality for V-JEPA2** — client-side image-to-video may yield clips V-JEPA2 finds noisy. | MEDIUM | Face close-up clip produced via MoviePy yields a peak around L_FFC/R_FFC at t≈5 s in the region table (smoke_test pass).                                                      | Harsh    |
| 5  | **HCP-MMP1 `.annot` fsaverage5 compatibility** — standard download is fsaverage7; we need fsaverage5-resampled. | MEDIUM | `atlases/lh.HCP-MMP1.annot` and `atlases/rh.HCP-MMP1.annot` load via `nibabel.freesurfer.read_annot` and the label array length equals 10242 per hemisphere.                  | Claude   |
| 6  | **Upstream API stability for `TribeModel.predict` signature**.                                              | LOW    | Pin upstream commit in `pyproject.toml` once verified; document the pin in an ADR.                                                                                            | upstream |

## Notes on impact ratings

- **HIGH** — blocks v1 launch or risks a known failure mode under load.
- **MEDIUM** — degrades demo quality but does not block launch.
- **LOW** — annoyance or maintenance hazard; no user-visible blast radius.

## Closing rows

A row is closed by linking either (a) the commit that demonstrates the verification check, or (b) the ADR that explicitly accepts the residual risk. Do not delete rows when closed — move them to a `## Closed` section at the bottom and keep the link, so the audit trail survives.
