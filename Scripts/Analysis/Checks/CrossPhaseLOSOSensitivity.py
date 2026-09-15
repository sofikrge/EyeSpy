"""Robustness check: does self-contribution to the intact reference map drive the effect?

The preregistered cross-phase NSS scores a participant's Mooney fixations against a
disambiguator reference map built from *all* participants — including that participant's
own disambiguation-phase fixations from the same trial. Because each image keeps one
disambiguator type per participant, the scored participant is inside the intact reference
and never inside the scrambled one, so the leakage is one-sided and lands on NSS_diff.
This script measures it rather than assuming it is small: it rescores every unit a second
time against a leave-one-subject-out reference, and reports both, plus a placebo that
drops a random *other* participant to separate self-similarity from the cost of a map
built from one fewer person.

The registered analysis is unchanged; this only reports what the alternative would give.

Reads:  data[_rep]/NSS_all_fixations_clean.parquet          (NSSExporter.py)
        analysesresults[_rep]/NSS[_suffix]/FixMaps_full.pkl  (NSS.py)
Writes: nothing (prints two tables)
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
from Settings import FIX_FILE, MASK_PPD, SIGMA, IMAGE_HEIGHT, IMAGE_WIDTH, MIN_VIEWINGS_PER_IMAGE_CROSS
from Scripts.Analysis.NSS import NSSPaths
import Scripts.Analysis.NSS.NSS as N

RANDOM_SEED = 0  # fixes which "other" participant the placebo drops


def _z(map_2d):
    """Z-normalised copy of a reference map, or None if it is missing/degenerate."""
    if map_2d is None:
        return None
    z, _, _ = N._z_normalize(np.asarray(map_2d, dtype=float))
    return z


def _loso_zmaps(fm, sigma):
    """participant id -> z-map of this reference with that participant left out."""
    if fm is None or not fm.get("subject"):
        return {}
    subs = fm["subject"]
    blurred, sum_blur = N._blur_subject_maps(
        N._stack_subject_maps(subs, IMAGE_HEIGHT, IMAGE_WIDTH), sigma)
    return {
        str(s["ParticipantID"]): _z(N._compute_loso(sum_blur, blurred, j))
        for j, s in enumerate(subs)
    }


def main():
    paths = NSSPaths.select()
    fixations = NSSPaths.filter_trial_set(pd.read_parquet(FIX_FILE), paths["TRIAL_SET"])
    with open(paths["FIXMAPS_PKL"], "rb") as f:
        FixMaps = pickle.load(f)["data"]

    fm_index = N._index_fixmaps(FixMaps)
    dy_off, dx_off = N._disk_offsets(SIGMA)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for fm in FixMaps:
        image_type = fm["image_type"]
        if not image_type.startswith("mooney_post_intact_"):
            continue
        awareness = image_type.split("mooney_post_intact_")[1]
        img, session = fm["img"], fm["condition"]  # note: "condition" holds the session

        df_group = N._group_fixations_for_image(fixations, img, session, "mooney_post_intact")
        df_group = df_group[df_group["awareness"] == awareness]
        coords, keys = N._coords_in_fixmaps_order(
            df_group, MASK_PPD, IMAGE_HEIGHT, IMAGE_WIDTH, return_keys=True)
        if len(coords) < int(MIN_VIEWINGS_PER_IMAGE_CROSS):
            continue

        refs = {"Intact": fm_index.get((img, session, "disamb_intact")),
                "Scrambled": fm_index.get((img, session, "disamb_not_intact"))}
        full = {k: _z(v["fixMapPerIm"]) if v else None for k, v in refs.items()}
        loso = {k: _loso_zmaps(v, SIGMA) for k, v in refs.items()}

        for (pid, trial), coords_unit in zip(keys, coords):
            row = {"Awareness": awareness, "Participant": pid, "Image": img, "Trial": trial}
            for ref in ("Intact", "Scrambled"):
                row[f"full_{ref}"] = N._nss_for_subject(full[ref], coords_unit, dy_off, dx_off)
                # leave the participant out only where they actually contributed
                row[f"loso_{ref}"] = N._nss_for_subject(
                    loso[ref].get(str(pid), full[ref]), coords_unit, dy_off, dx_off)
                row[f"self_in_{ref}"] = str(pid) in loso[ref]

            # Placebo: drop a random OTHER contributor to the intact map, isolating how
            # much of the LOSO shift is just "one fewer person in the map".
            others = [q for q in loso["Intact"] if q != str(pid)]
            row["placebo_Intact"] = (
                N._nss_for_subject(loso["Intact"][str(rng.choice(others))],
                                   coords_unit, dy_off, dx_off)
                if others else np.nan)
            rows.append(row)

    d = pd.DataFrame(rows)
    d["full_diff"] = d["full_Intact"] - d["full_Scrambled"]
    d["loso_diff"] = d["loso_Intact"] - d["loso_Scrambled"]

    print(f"\nScored units: {len(d)}   "
          f"participant inside their Intact reference: {100 * d.self_in_Intact.mean():.1f}%   "
          f"inside their Scrambled reference: {100 * d.self_in_Scrambled.mean():.1f}%")

    print("\nPer-unit means, registered (full map) vs leave-one-subject-out:")
    for awareness, sub in d.groupby("Awareness"):
        print(f"  {awareness}  (n = {len(sub)})")
        for label, a, b in (("Intact", "full_Intact", "loso_Intact"),
                            ("Scrambled", "full_Scrambled", "loso_Scrambled"),
                            ("NSS_diff", "full_diff", "loso_diff")):
            print(f"      {label:<10} registered {sub[a].mean():+.4f}   "
                  f"LOSO {sub[b].mean():+.4f}   change {sub[b].mean() - sub[a].mean():+.4f}")

    print("\nParticipant-level NSS_diff (the unit the model and the violin plot use):")
    means = {}
    for awareness, sub in d.groupby("Awareness"):
        per_p = sub.groupby("Participant")[["full_diff", "loso_diff"]].mean()
        means[awareness] = (per_p.full_diff.mean(), per_p.loso_diff.mean())
        print(f"  {awareness:<22} n = {len(per_p):>3}   "
              f"registered {means[awareness][0]:+.4f}   LOSO {means[awareness][1]:+.4f}")
    if len(means) == 2:
        (a1, l1), (a2, l2) = means.values()
        print(f"  {'group difference':<22}       "
              f"registered {a1 - a2:+.4f}   LOSO {l1 - l2:+.4f}   "
              "<- the registered hypothesis is the interaction")

    placebo = d.dropna(subset=["placebo_Intact"])
    print("\nPlacebo — is the shift self-similarity, or just a map with one fewer person?")
    print(f"  Intact, full reference             {placebo.full_Intact.mean():+.4f}")
    print(f"  Intact, dropping the scored person {placebo.loso_Intact.mean():+.4f}   "
          f"change {placebo.loso_Intact.mean() - placebo.full_Intact.mean():+.4f}")
    print(f"  Intact, dropping a random other    {placebo.placebo_Intact.mean():+.4f}   "
          f"change {placebo.placebo_Intact.mean() - placebo.full_Intact.mean():+.4f}")
    print(f"  self-similarity alone              "
          f"{placebo.loso_Intact.mean() - placebo.placebo_Intact.mean():+.4f}")


if __name__ == "__main__":
    main()
