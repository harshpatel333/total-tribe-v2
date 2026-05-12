# Risks

Open verification gates from `docs/SPIKE_FINDINGS.md` (§ "What still needs verifying"), plus the LLaMA approval blocker. This is a living register; close rows by linking to the verifying commit/PR/ADR.

## Open

| #  | Risk                                                                                                       | Impact | Verification gate (closes when…)                                                                                                                                              |
| -- | ---------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **LLaMA-3.2-3B approval pending for the project maintainer's HuggingFace account.** Blocks text-modality real-load testing. | HIGH   | Meta approves access on <https://huggingface.co/meta-llama/Llama-3.2-3B>; `hf auth whoami` shows the account; a smoke `TribeModel.from_pretrained` call (with text enabled) succeeds with weights. |
| 4  | **`CreateVideosFromImages` quality for V-JEPA2** — client-side image-to-video may yield clips V-JEPA2 finds noisy or inconsistent with the training distribution. | MEDIUM | Face close-up clip produced via MoviePy yields a peak around `L_FFC`/`R_FFC` (fusiform face area) at `t ≈ 5 s` in the region table. The deploy-time smoke test with a synthetic face landed on `R_MT` (motion area) instead — sensible for a static loop but not a face-area peak. Re-test with a richer face clip before claiming this is closed. |
| 6  | **Upstream API stability for `TribeModel.predict` signature**.                                              | LOW    | Pin upstream commit in `pyproject.toml` once verified; document the pin in an ADR.                                                                                            |

## Closed

| #  | Risk | Closed by |
| -- | ---- | --------- |
| 2  | **VRAM peak under real load** — accounting predicted ~23 GB on a 24 GB card. | First deploy smoke test (commit `14178e6`): image peak 14014 MiB, video peak 14006 MiB, audio peak ~4.5 GB; all well under the 24 GB cap. With LLaMA enabled the budget rises ~7 GB but stays under the cap. |
| 3  | **`get_events_dataframe` actual column set** — docstring claims `type`, `filepath`, `start`, `duration`, `timeline`, `subject`; we had not run it. | Invoked end-to-end on the deployed UI for image, audio, and video paths (see `docs/images/0{1..3}-*.png`); all three return populated predictions with the expected `(T, 20484)` shape. |
| 5  | **HCP-MMP1 `.annot` fsaverage5 compatibility** — standard Figshare download is fsaverage7; we needed fsaverage5-resampled. | Switched to GOBS-derived labels in `scripts/fetch_atlases.py` (commit `7ee0de3`). `nibabel.freesurfer.read_annot` returns 10242 labels per hemisphere; verified at runtime via the deployed region table. |

## Notes on impact ratings

- **HIGH** — blocks v1 launch or risks a known failure mode under load.
- **MEDIUM** — degrades demo quality but does not block launch.
- **LOW** — annoyance or maintenance hazard; no user-visible blast radius.

## Closing rows

A row is closed by linking either (a) the commit that demonstrates the verification check, or (b) the ADR that explicitly accepts the residual risk. Do not delete rows when closed — move them to the `## Closed` table above and keep the link, so the audit trail survives.
