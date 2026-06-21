"""
DiagnoseParticipantCoverage.py
-------------------------------
Diagnoses why a participant may have very few observations in the cross-phase NSS
ranking (unconscious_unaware). Runs three checks in sequence:

  1. How many mooney_post_intact + unconscious_unaware images does this participant
     have in the fixations parquet? (cross-phase only scores post-intact Mooneys)
  2. For each of those images, how many *other* UU participants also appear?
     Images where this is 0 will be dropped by MIN_SUBJ_PER_IMAGE_CROSS.
  3. What images did this participant actually get scored on in CrossResults?

Requires:
  data/NSS_all_fixations_clean.parquet
  analysesresults/NSS/NSS_crossphase_descriptives.pkl
"""

from pathlib import Path
import pickle
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARTICIPANT_ID       = "108"                  # participant to inspect (as string)
MIN_SUBJ_PER_IMAGE   = 2                    # must match NSS.py setting
FIX_FILE             = Path("data/NSS_all_fixations_clean.parquet")
CROSS_CACHE          = Path("analysesresults/NSS/NSS_crossphase_descriptives.pkl")
TARGET_AWARENESS     = "unconscious_unaware"
TARGET_IMAGE_TYPE    = "mooney_post_intact"
# ─────────────────────────────────────────────────────────────────────────────

fix   = pd.read_parquet(FIX_FILE)
cross = pickle.load(open(CROSS_CACHE, "rb"))
cross = cross["data"] if isinstance(cross, dict) else cross

SEP = "=" * 60

# ── CHECK 1: Images this participant has in the parquet ───────────────────────
print(f"\n{SEP}")
print(f"CHECK 1 — mooney_post_intact + {TARGET_AWARENESS} fixations in parquet")
print(f"          (cross-phase NSS ignores mooney_post_scrambled entirely)")
print(SEP)

p_fix = fix[
    (fix["participant"].astype(str) == PARTICIPANT_ID) &
    (fix["image_type"] == TARGET_IMAGE_TYPE) &
    (fix["awareness"] == TARGET_AWARENESS)
]

p_images = p_fix["ImageName"].unique()
per_image = p_fix.groupby("ImageName").size().rename("n_fixations")

if p_fix.empty:
    print(f"  !! NO fixations found for participant {PARTICIPANT_ID} in this condition.")
    print(f"     Check that their awareness label and image_type are assigned correctly.")
else:
    print(f"  Participant {PARTICIPANT_ID} has {len(p_images)} qualifying images "
          f"and {len(p_fix)} fixations total.\n")
    print(per_image.to_string())

# ── CHECK 2: Co-participant coverage per image ────────────────────────────────
print(f"\n{SEP}")
print(f"CHECK 2 — other {TARGET_AWARENESS} participants per image")
print(f"          (images with 0 others fail the MIN_SUBJ={MIN_SUBJ_PER_IMAGE} threshold → dropped)")
print(SEP)

all_uua = fix[
    (fix["image_type"] == TARGET_IMAGE_TYPE) &
    (fix["awareness"] == TARGET_AWARENESS)
]

if p_fix.empty:
    print("  Skipped — participant has no qualifying fixations (see Check 1).")
else:
    rows = []
    for img in sorted(p_images):
        others = all_uua[
            (all_uua["ImageName"] == img) &
            (all_uua["participant"].astype(str) != PARTICIPANT_ID)
        ]["participant"].nunique()
        will_be_scored = others >= (MIN_SUBJ_PER_IMAGE - 1)  # need ≥1 other (total ≥2)
        rows.append({
            "ImageName":    img,
            "n_others":     others,
            "will_be_scored": "YES" if will_be_scored else "NO  <-- dropped",
        })

    df_cov = pd.DataFrame(rows).sort_values("n_others")
    n_pass = (df_cov["n_others"] >= MIN_SUBJ_PER_IMAGE - 1).sum()
    n_fail = len(df_cov) - n_pass

    print(f"  {n_pass}/{len(df_cov)} images pass the threshold, {n_fail} are dropped.\n")
    print(df_cov.to_string(index=False))

# ── CHECK 3: What CrossResults actually scored for this participant ────────────
print(f"\n{SEP}")
print(f"CHECK 3 — images participant {PARTICIPANT_ID} appears in inside CrossResults")
print(f"          (ground truth: what the NSS cache actually computed)")
print(SEP)

scored = []
for entry in cross["image"]:
    if entry.get("awareness") != TARGET_AWARENESS:
        continue
    for subj in entry.get("subject", []):
        if subj.get("ParticipantID", "").split("_t")[0] == PARTICIPANT_ID:
            scored.append({
                "ImageName":     entry["img"],
                "ParticipantID": subj["ParticipantID"],
                "NSS_intact":    subj.get("NSS_intact"),
                "NSS_scrambled": subj.get("NSS_scrambled"),
                "NSS_diff":      subj.get("NSS_diff"),
            })

if not scored:
    print(f"  !! Participant {PARTICIPANT_ID} was not scored on ANY image.")
    print(f"     All their images likely failed the MIN_SUBJ threshold (see Check 2).")
else:
    df_scored = pd.DataFrame(scored)
    print(f"  Participant {PARTICIPANT_ID} was scored on {len(df_scored)} image-trial(s).\n")
    print(df_scored.to_string(index=False))

print(f"\n{SEP}\nDone.\n")
