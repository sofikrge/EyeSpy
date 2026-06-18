# NSS_Debug.py
# Run this standalone to diagnose dropped images, cell sizes, and data integrity.

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

NSS_DIR   = Path("analysesresults/NSS")
FIX_FILE  = Path("data/NSS_all_fixations_clean.parquet")

MIN_SUBJ_PER_IMAGE_NSS   = 2
MIN_SUBJ_PER_IMAGE_CROSS = 2

# ── Load ──────────────────────────────────────────────────────────────────────
fixations   = pd.read_parquet(FIX_FILE)
FixMaps     = pickle.load(open(NSS_DIR / "FixMaps_full.pkl",              "rb"))["data"]
NSSResults  = pickle.load(open(NSS_DIR / "NSS_WithinPhase.pkl",           "rb"))["data"]
CrossResults= pickle.load(open(NSS_DIR / "NSS_crossphase_descriptives.pkl","rb"))["data"]

print("=" * 60)
print("1. FIXMAP INVENTORY")
print("=" * 60)
from collections import Counter
fm_types = Counter((fm["image_type"], fm["condition"]) for fm in FixMaps)
for k, v in sorted(fm_types.items()):
    print(f"  {k[0]:45s}  session={k[1]}  n_images={v}")

# ── Within-phase: dropped breakdown ───────────────────────────────────────────
print("\n" + "=" * 60)
print("2. WITHIN-PHASE: DROPPED IMAGES BREAKDOWN")
print("=" * 60)
dropped = []
for fm in FixMaps:
    n = len(fm.get("subject", []))
    if n < MIN_SUBJ_PER_IMAGE_NSS:
        dropped.append({
            "image_type": fm["image_type"],
            "condition":  fm["condition"],
            "n_subjects": n,
        })
df_drop = pd.DataFrame(dropped)
if df_drop.empty:
    print("  No images dropped.")
else:
    print(df_drop.groupby(["image_type", "condition"])["n_subjects"]
              .agg(["count", "mean", "max"])
              .rename(columns={"count": "n_dropped", "mean": "avg_subj", "max": "max_subj"})
              .to_string())

# ── Within-phase: subject count distribution ──────────────────────────────────
print("\n" + "=" * 60)
print("3. WITHIN-PHASE: SUBJECT COUNT DISTRIBUTION PER IMAGE TYPE")
print("=" * 60)
rows = []
for fm in FixMaps:
    rows.append({"image_type": fm["image_type"], "condition": fm["condition"], "n_subjects": len(fm.get("subject", []))})
df_fm = pd.DataFrame(rows)
print(df_fm.groupby(["image_type", "condition"])["n_subjects"]
          .describe(percentiles=[.25, .5, .75])
          .round(1).to_string())

# ── Cross-phase: per-awareness cell sizes ─────────────────────────────────────
print("\n" + "=" * 60)
print("4. CROSS-PHASE: SUBJECTS PER (IMAGE × SESSION × AWARENESS)")
print("=" * 60)
cross_rows = []
for rec in CrossResults["image"]:
    cross_rows.append({
        "awareness":   rec["awareness"],
        "condition":   rec["condition"],
        "n_subjects":  len(rec.get("subject", [])),
        "valid_intact":   np.isfinite(rec.get("NSS_intact_img",   float("nan"))),
        "valid_scrambled":np.isfinite(rec.get("NSS_scrambled_img",float("nan"))),
    })
df_cross = pd.DataFrame(cross_rows)
print(df_cross.groupby(["awareness", "condition"]).agg(
    n_images      =("n_subjects","count"),
    avg_subj      =("n_subjects","mean"),
    min_subj      =("n_subjects","min"),
    n_valid_intact=("valid_intact","sum"),
    n_valid_scram =("valid_scrambled","sum"),
).round(1).to_string())

# ── Cross-phase: images with 0 subjects ───────────────────────────────────────
print("\n" + "=" * 60)
print("5. CROSS-PHASE: IMAGES WITH 0 SUBJECTS (fully dropped)")
print("=" * 60)
zero = df_cross[df_cross["n_subjects"] == 0]
if zero.empty:
    print("  None.")
else:
    print(zero.groupby(["awareness","condition"]).size().to_string())

# ── Eccentricity lookup coverage ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ECCENTRICITY LOOKUP COVERAGE")
print("=" * 60)
disamb_lookup = {}
for img_data in NSSResults["image"]:
    if img_data.get("image_type") == "disamb_intact":
        for subj in img_data["subject"]:
            if "ParticipantID" in subj:
                disamb_lookup[(subj["ParticipantID"], img_data["img"], img_data["condition"])] = subj["NSSSimPerSubj"]

print(f"  Disambiguation eccentricity scores available: {len(disamb_lookup)}")

# Check coverage against cross-phase export keys
n_matched, n_missing = 0, 0
for rec in CrossResults["image"]:
    for subj in rec.get("subject", []):
        pid = subj.get("ParticipantID","").split("_t")[0]
        key = (pid, rec["img"], rec["condition"])
        if key in disamb_lookup:
            n_matched += 1
        else:
            n_missing += 1
print(f"  Cross-phase rows matched to eccentricity score: {n_matched}")
print(f"  Cross-phase rows WITHOUT eccentricity score:    {n_missing}  ← will be NaN in NSS_Corrected")

# ── Fixation counts by awareness ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. RAW FIXATION COUNTS BY IMAGE_TYPE × SESSION × AWARENESS")
print("=" * 60)
print(fixations.groupby(["image_type","session","awareness"])
      .size().reset_index(name="n_fixations").to_string(index=False))

print("\nDone.")