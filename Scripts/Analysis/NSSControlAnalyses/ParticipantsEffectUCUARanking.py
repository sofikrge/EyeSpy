"""
ParticipantsEffectUCUARanking.py
---------------------------------
Ranks all participants in the unconscious_unaware condition by their contribution to the
cross-phase NSS effect: Mooney fixations overlap MORE with where others looked during the
SCRAMBLED than the INTACT disambiguation (i.e. NSS_scrambled > NSS_intact → NSS_diff < 0).

Participants are sorted by mean NSS_diff (intact − scrambled) ascending: most-negative
at the top drove the effect hardest; positive values worked against it.

Requires: analysesresults/NSS/NSS_crossphase_descriptives.pkl  (produced by NSS.py)
"""

from pathlib import Path
import pickle
import pandas as pd

NSS_DIR         = Path("analysesresults/NSS")
CROSS_CACHE     = NSS_DIR / "NSS_crossphase_descriptives.pkl"
TARGET_AWARENESS = "unconscious_unaware"

# ── Load cross-phase results ───────────────────────────────────────────────────
with open(CROSS_CACHE, "rb") as f:
    cache = pickle.load(f)
results = cache["data"] if isinstance(cache, dict) else cache

# ── Flatten per-subject, per-image scores for the target awareness group ───────
rows = []
for entry in results["image"]:
    if entry.get("awareness") != TARGET_AWARENESS:
        continue
    for subj in entry.get("subject", []):
        pid = subj.get("ParticipantID", "")
        rows.append({
            "participant":   pid.split("_t")[0],  # strip trial suffix (e.g. "42_t3" → "42")
            "image":         entry["img"],
            "NSS_intact":    subj.get("NSS_intact"),
            "NSS_scrambled": subj.get("NSS_scrambled"),
            "NSS_diff":      subj.get("NSS_diff"),  # intact − scrambled; negative drives the effect
        })

df = pd.DataFrame(rows)

# ── Aggregate per participant (mean across images/trials, ignoring NaN) ────────
agg = (
    df.groupby("participant")
    .agg(
        n_obs         = ("NSS_diff", "count"),   # valid (non-NaN) image observations
        mean_intact   = ("NSS_intact",    "mean"),
        mean_scrambled= ("NSS_scrambled", "mean"),
        mean_diff     = ("NSS_diff",      "mean"),
    )
    .sort_values("mean_diff", ascending=True)    # most-negative (effect driver) → top
    .reset_index()
)
agg.index += 1  # 1-based rank

# ── Print ranked table ─────────────────────────────────────────────────────────
n_participants = len(agg)
group_mean_diff = df["NSS_diff"].mean()

print(f"\nCross-phase NSS ranking - {TARGET_AWARENESS}  (N={n_participants} participants)")
print(f"Effect: NSS_scrambled > NSS_intact  →  negative NSS_diff drives the effect")
print(f"Group mean NSS_diff = {group_mean_diff:.4f}\n")

header = f"{'Rank':>4}  {'Participant':<14}  {'n_obs':>5}  {'NSS_intact':>10}  {'NSS_scrambled':>13}  {'NSS_diff':>9}"
print(header)
print("-" * len(header))

for rank, row in agg.iterrows():
    # mark participants whose NSS_diff pulls in the effect direction (< 0)
    marker = " <--" if row["mean_diff"] < 0 else ""
    print(
        f"{rank:>4}  {row['participant']:<14}  {row['n_obs']:>5.0f}  "
        f"{row['mean_intact']:>10.4f}  {row['mean_scrambled']:>13.4f}  "
        f"{row['mean_diff']:>9.4f}{marker}"
    )

n_driving = (agg["mean_diff"] < 0).sum()
print(f"\n{n_driving}/{n_participants} participants show NSS_diff < 0 (scrambled > intact).")