# ADR-0004: HCP-MMP1 (Glasser et al. 2016) for cortical parcellation

- **Status:** Accepted
- **Date:** 2026-05-11

## Context

TRIBE v2 predictions are 20,484 cortical vertices on `fsaverage5` — too granular to display directly as a region table. The UI needs an atlas that maps vertices into named parcels with **interpretable, layperson-readable** names, because the audience is researchers and educators, not neuroanatomists.

The candidates considered, in rough order of fame:

- **HCP-MMP1 (Glasser et al. 2016)** — 180 parcels per hemisphere, defined multimodally (function + connectivity + cytoarchitecture). Names like `FFC` (Fusiform Face Complex), `V1`, `A1`, `MT`, `LIPv` that are widely used in the cognitive-neuroscience literature.
- **Schaefer 2018 / Yeo 7- or 17-network parcellations** — well-validated, but parcel names are mostly numeric (`7Networks_LH_Vis_1`, etc.) and require an external lookup to recover function.
- **Destrieux 2010** — purely anatomical (sulcus/gyrus names), no functional information. Hard to talk about for stimulus-driven activation.
- **Custom hand labeling.** Out of scope for v1.

## Decision

Use **HCP-MMP1** as the parcellation backing the region interpreter. Ship FreeSurfer `.annot` files (LH and RH) resampled to `fsaverage5` in `atlases/`, and use `nibabel.freesurfer.read_annot` to map every vertex to its parcel.

Combine HCP-MMP1 parcel names with Neurosynth-derived cognitive terms (see `docs/INTERPRETATION.md`) so the UI can display, for example, `L_FFC → faces, recognition, fusiform`.

## Consequences

**Positive:**

- Parcel names communicate function directly to a literate reader. `L_FFC` reads better than `7Networks_LH_Vis_4`.
- Pairs naturally with Neurosynth: the parcel centroid is enough to query Neurosynth for term associations, and the parcel name is a stable join key.
- 360 parcels (180 × 2 hemispheres) is the right granularity for a top-K display — not so coarse that everything maps to the same place, not so fine that the table looks like noise.

**Negative:**

- `atlases/` must ship `.annot` files (~few MB). Acceptable.
- HCP-MMP1's published resampling is to `fsaverage7`; we depend on a community-resampled fsaverage5 version. Tracked as a verification risk in `docs/RISKS.md`.
- The parcellation is **group-average**. Individual subjects have functional variability that an atlas cannot capture. This is consistent with the project's "not for individual subjects" disclaimer in `README.md`.

## Alternatives considered

- **Schaefer/Yeo.** Rejected — parcels are reliable but the numeric names hurt UX, and the network labels are coarser than what the UI wants to show.
- **Destrieux.** Rejected — anatomical only. Telling a user "the predicted activation peaks in the lateral occipital sulcus" is technically true but unhelpful for stimulus-driven interpretation, where the same coordinates would much more usefully be reported as "Fusiform Face Complex".
- **Custom manual labeling of regions we expect to activate.** Rejected — out of scope, brittle, and not principled enough to be defensible in a research/education context.
