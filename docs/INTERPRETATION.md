# Interpretation

This document describes how raw TRIBE v2 predictions become a human-readable region table.

## Region lookup pipeline

1. **Prediction.** `TribeModel.predict()` returns `preds: np.ndarray` with shape `(T, 20484)` on the `fsaverage5` cortical surface, sampled at 1 Hz.

2. **Split hemispheres.** `fsaverage5` has 10242 vertices per hemisphere. We split:
   - `lh = preds[:, :10242]`
   - `rh = preds[:, 10242:]`

3. **Map vertices to parcels.** HCP-MMP1 ships as a FreeSurfer `.annot` per hemisphere (`lh.HCP-MMP1.annot`, `rh.HCP-MMP1.annot`), resampled to `fsaverage5`. The annot files give every vertex an integer parcel label (0..180 per hemisphere, with `0` = medial wall / unassigned). The labels are read once at startup with `nibabel.freesurfer.read_annot`.

4. **Aggregate per parcel per timepoint.** For each timepoint `t` and each parcel `p`, we compute the mean activation across all vertices in `p`. This yields a `(T, 360)` parcel-activation matrix (180 LH + 180 RH).

5. **Top-K.** Per timepoint, return the K parcels with the highest absolute mean activation. K defaults to 10 and is configurable via the UI.

6. **Cognitive term lookup.** Each parcel name (e.g. `L_FFC`, `R_V1`) is looked up against `atlases/parcel_lookup.json` to retrieve a short list of Neurosynth meta-analytic terms describing the parcel's typical functional role.

## `atlases/parcel_lookup.json` schema

```json
{
  "L_FFC": {
    "terms": ["faces", "recognition", "fusiform"],
    "score": 0.87
  },
  "R_FFC": {
    "terms": ["faces", "recognition"],
    "score": 0.85
  }
}
```

Field semantics:

- **Key** — HCP-MMP1 parcel name in the form `<hemi>_<parcel>` where `<hemi>` is `L` or `R` and `<parcel>` is the Glasser et al. 2016 short label (e.g. `FFC` = Fusiform Face Complex, `V1` = primary visual cortex, `A1` = primary auditory cortex).
- **`terms`** — A list of Neurosynth meta-analytic terms most strongly associated with the parcel's coordinates, ordered by association strength.
- **`score`** — A normalized 0..1 association strength for the top term (informational; used in the UI for a confidence indicator).

## Hand-curated fallback

For v1, `atlases/parcel_lookup.json` ships with **~50 hand-curated parcels** covering the regions most likely to activate under naturalistic stimuli (early visual, ventral temporal face/object/scene complex, primary and association auditory, language network, default mode hubs). This is enough to make the demo legible for the expected use cases.

Full coverage of all 360 parcels can be generated offline with `scripts/build_neurosynth_lookup.py`, which queries NiMARE / Neurosynth against the parcel centroid coordinates and emits a parcel-level term ranking. The script is **optional** — it is not part of the container build, requires its own internet-fetched data, and is intended to be run once on a workstation and the resulting JSON committed.
