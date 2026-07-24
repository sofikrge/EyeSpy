"""Why does one participant have so few observations in the cross-phase NSS?

Set PARTICIPANT_ID below and run. Three checks, narrowing from "is the data there
at all" to "did the drop rules remove it":

  1. How many mooney_post_intact + unconscious_unaware fixations does this
     participant have in the parquet? (cross-phase only scores post-intact Mooneys)
  2. For each of those images, how many *other* UU participants also appear?
     Images where this is 0 get dropped by MIN_SUBJ_PER_IMAGE_CROSS.
  3. Which images did the participant actually get scored on?

Reads:  data/NSS_all_fixations_clean.parquet                        (NSSExporter.py)
        analysesresults/NSS_<mode>/NSS_crossphase_descriptives.pkl  (NSS.py)
"""

from pathlib import Path
import pickle
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for the imports below
from Scripts.Analysis.NSS import NSSPaths
from Settings import FIX_FILE, MIN_SUBJ_PER_IMAGE_CROSS as MIN_SUBJ_PER_IMAGE
_P = NSSPaths.select()  # prompt or $MOONEY_SPLIT -> per-mode folder

# --- CONFIG
PARTICIPANT_ID    = "108"   # participant to inspect (as string)
CROSS_CACHE       = _P["CROSS_PKL"]
TARGET_AWARENESS  = "unconscious_unaware"
TARGET_IMAGE_TYPE = "mooney_post_intact"

# --- Load
fix   = NSSPaths.filter_trial_set(pd.read_parquet(FIX_FILE), _P["TRIAL_SET"])  # match NSS.py's trial set
cross = pickle.load(open(CROSS_CACHE, "rb"))
cross = cross["data"] if isinstance(cross, dict) else cross

SEP = "=" * 60

# --- CHECK 1: Images this participant has in the parquet
print(f"\n{SEP}")
print(f"CHECK 1 - mooney_post_intact + {TARGET_AWARENESS} fixations in parquet")
print("          (cross-phase NSS ignores mooney_post_scrambled entirely)")
print(SEP)

p_fix = fix[
    (fix["participant"].astype(str) == PARTICIPANT_ID) &
    (fix["image_type"] == TARGET_IMAGE_TYPE) &
    (fix["awareness"] == TARGET_AWARENESS)
]

p_images = p_fix["ImageName"].unique()
per_image = p_fix.groupby("ImageName").size().rename("n_fixations")

if p_fix.empty:
    print(f"  WARNING: no fixations found for participant {PARTICIPANT_ID} in this condition.")
    print("           Check that their awareness label and image_type are assigned correctly.")
else:
    print(f"  Participant {PARTICIPANT_ID} has {len(p_images)} qualifying images "
          f"and {len(p_fix)} fixations total.\n")
    print(per_image.to_string())

# --- CHECK 2: Co-participant coverage per image
print(f"\n{SEP}")
print(f"CHECK 2 - other {TARGET_AWARENESS} participants per image")
print(f"          (images with 0 others fail the MIN_SUBJ={MIN_SUBJ_PER_IMAGE} threshold -> dropped)")
print(SEP)

all_uua = fix[
    (fix["image_type"] == TARGET_IMAGE_TYPE) &
    (fix["awareness"] == TARGET_AWARENESS)
]

if p_fix.empty:
    print("  Skipped - participant has no qualifying fixations (see Check 1).")
else:
    rows = []
    for img in sorted(p_images):
        others = all_uua[
            (all_uua["ImageName"] == img) &
            (all_uua["participant"].astype(str) != PARTICIPANT_ID)
        ]["participant"].nunique()
        will_be_scored = others >= (MIN_SUBJ_PER_IMAGE - 1)  # need >=1 other (total >=2)
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

# --- CHECK 3: What CrossResults actually scored for this participant
print(f"\n{SEP}")
print(f"CHECK 3 - images participant {PARTICIPANT_ID} appears in inside CrossResults")
print("          (ground truth: what the NSS cache actually computed)")
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
    print(f"  WARNING: participant {PARTICIPANT_ID} was not scored on any image.")
    print("           All their images likely failed the MIN_SUBJ threshold (see Check 2).")
else:
    df_scored = pd.DataFrame(scored)
    print(f"  Participant {PARTICIPANT_ID} was scored on {len(df_scored)} image-trial(s).\n")
    print(df_scored.to_string(index=False))

print(f"\n{SEP}\nDone.\n")
