"""
NSSFixationCounts.py
--------------------
How many fixations go into the cross-phase NSS calculation?

The cross-phase NSS scores each participant's Post-Intact Mooney fixations (per
trial) against the disambiguation-phase reference maps (intact + scrambled). Two
sets of fixations therefore enter the calculation, and this script counts both,
each split by awareness and intact vs scrambled:

  TABLE 1 — Reference-map fixations
      Disambiguation-phase fixations that BUILD the reference maps, restricted to
      the maps actually used in scoring (i.e. images that survived the cross-phase
      drop rules). "Intact"/"Scrambled" = the disambiguator image type
      (disamb_intact / disamb_not_intact). Reference maps are pooled per session in
      NSS.py; the awareness split here shows their composition.

  TABLE 2 — Scored Post-Intact Mooney fixations
      The DV: each Mooney fixation read off both reference maps. "Intact"/"Scrambled"
      = which reference it was scored against. Driven by the results pickle so every
      NSS.py drop rule (MIN_SUBJ_PER_IMAGE_CROSS, participant x trial unit, permissive
      NaN policy) is honored. A fixation scored against both references is counted in
      both columns; the distinct total is printed above the table.

Run from the project root:
    uv run python3 Scripts/Analysis/NSS/NSSFixationCounts.py

Requires:
    data/NSS_all_fixations_clean.parquet                  (NSSExporter.py)
    analysesresults/NSS/NSS_crossphase_descriptives.pkl   (NSS.py)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# === CONFIG (must match NSS.py) ===
FIX_FILE     = Path("data/NSS_all_fixations_clean.parquet")
CROSS_PKL    = Path("analysesresults/NSS/NSS_crossphase_descriptives.pkl")
IMAGE_HEIGHT = 600
IMAGE_WIDTH  = 800
MASK_PPD     = 48.55

UNIT_KEYS  = ["participant", "ImageName", "session", "trial_number", "awareness"]
DISAMB_MAP = {"disamb_intact": "Intact", "disamb_not_intact": "Scrambled"}


def _in_bounds(x_deg, y_deg):
    """1-based pixel transform + bounds test, identical to NSS.py."""
    w = np.sign(x := IMAGE_WIDTH  / 2.0 + x_deg * MASK_PPD) * np.floor(np.abs(x) + 0.5)
    h = np.sign(y := IMAGE_HEIGHT / 2.0 + y_deg * MASK_PPD) * np.floor(np.abs(y) + 0.5)
    return (h >= 1) & (h <= IMAGE_HEIGHT) & (w >= 1) & (w <= IMAGE_WIDTH)


def load_inbounds_fixations():
    """All finite, in-bounds fixations (the only ones that land on a map)."""
    fix = pd.read_parquet(FIX_FILE)
    for c in ("x_deg_centered", "y_deg"):
        fix[c] = pd.to_numeric(fix[c], errors="coerce")
    fix = fix.dropna(subset=["x_deg_centered", "y_deg"])
    return fix[_in_bounds(fix["x_deg_centered"].to_numpy(), fix["y_deg"].to_numpy())]


def load_scored_units():
    """Ground-truth scored Mooney units (past every drop rule) from the pickle.

    One row per (participant, image, session, trial, awareness) with flags for which
    reference each unit produced a finite NSS against, plus the set of (image,
    session) reference maps that were actually used in scoring.
    """
    cross = pickle.load(open(CROSS_PKL, "rb"))["data"]
    rows, used_maps = [], set()
    for entry in cross["image"]:
        if not entry.get("subject"):
            continue
        used_maps.add((str(entry["img"]), str(entry["condition"])))
        for subj in entry["subject"]:
            pid, trial = subj["ParticipantID"].split("_t")
            rows.append({
                "participant": pid, "ImageName": str(entry["img"]),
                "session": str(entry["condition"]), "trial_number": trial,
                "awareness": subj["awareness"],
                "Intact": np.isfinite(subj["NSS_intact"]),
                "Scrambled": np.isfinite(subj["NSS_scrambled"]),
            })
    return pd.DataFrame(rows).drop_duplicates(UNIT_KEYS), used_maps


def _print_table(df, count_col="n_fixations"):
    df = df.sort_values(["awareness", "ReferenceMap"])
    print(df.to_string(index=False), "\n")


def reference_map_table(fix, used_maps):
    """Disamb fixations building the used reference maps, by awareness x type."""
    d = fix[fix["image_type"].isin(DISAMB_MAP)].copy()
    key = list(zip(d["ImageName"].astype(str), d["session"].astype(str)))
    d = d[pd.Series(key, index=d.index).isin(used_maps)]
    d["ReferenceMap"] = d["image_type"].map(DISAMB_MAP)
    return (d.groupby(["awareness", "ReferenceMap"], observed=True)
              .size().rename("n_fixations").reset_index())


def scored_mooney_table(fix, units):
    """Scored Post-Intact Mooney fixations, by awareness x reference scored against."""
    mooney = (fix[fix["image_type"] == "mooney_post_intact"]
              .groupby(UNIT_KEYS, observed=True).size().rename("n_fixations").reset_index())
    scored = mooney.merge(units, on=UNIT_KEYS, how="inner")
    long = scored.melt(id_vars=["awareness", "n_fixations"],
                       value_vars=["Intact", "Scrambled"],
                       var_name="ReferenceMap", value_name="hit")
    long = long[long["hit"]]
    table = (long.groupby(["awareness", "ReferenceMap"], observed=True)["n_fixations"]
                 .sum().rename("n_fixations").reset_index())
    return table, scored


def main():
    fix = load_inbounds_fixations()
    units, used_maps = load_scored_units()

    print("\n" + "=" * 70)
    print("TABLE 1 — Reference-map fixations (disambiguation phase)")
    print(f"  build the {len(used_maps)} (image x session) reference maps used in scoring")
    ref = reference_map_table(fix, used_maps)
    print(f"  {int(ref['n_fixations'].sum()):,} total disambiguation fixations\n")
    _print_table(ref)

    print("=" * 70)
    print("TABLE 2 — Scored Post-Intact Mooney fixations (the DV)")
    scored_tbl, scored = scored_mooney_table(fix, units)
    n_fix = int(scored["n_fixations"].sum())
    print(f"  {n_fix:,} distinct fixations  |  {len(scored):,} participant-trial units  "
          f"|  {scored['ImageName'].nunique()} images  |  {scored['participant'].nunique()} participants")
    print("  (each scored against both references → counted in both columns)\n")
    _print_table(scored_tbl)


if __name__ == "__main__":
    main()
