"""
ImagePerParticipant.py
----------------------
Counts how many image entries each participant contributes to the cross-phase
NSS analysis, broken down by awareness state and reference map (Intact / Scrambled).

Each row in the CSV is one scoring event (image x trial), so an image seen on two
trials counts as two entries -- both are valid data points.

Prints one easy-to-read table per awareness state, flagging any participant with
fewer than MIN_IMAGES entries (per reference map) with a "<--" marker.

Requires: analysesresults/NSS/NSS_CrossPhase_LongFormat.csv  (produced by NSS.py)
"""

from pathlib import Path
import pandas as pd

CROSS_CSV  = Path("analysesresults/NSS") / "NSS_CrossPhase_LongFormat.csv"
MIN_IMAGES = 13  # participants below this (per reference map) are flagged

# ── Load and count image entries per participant × awareness × reference map ────
df = pd.read_csv(CROSS_CSV)

counts = (
    df.groupby(["Awareness", "Participant", "ReferenceMap"])["Image"]
    .size()  # count rows (scoring events), so trial repeats count separately
    .unstack("ReferenceMap", fill_value=0)  # one column per reference map
    .sort_index()
)

ref_maps = list(counts.columns)  # e.g. ["Intact", "Scrambled"]

# ── Print one table per awareness state ─────────────────────────────────────────
for awareness, group in counts.groupby(level="Awareness"):
    group = group.droplevel("Awareness")

    print(f"\n{awareness}  (N={len(group)} participants, flagging < {MIN_IMAGES} entries)")
    header = f"{'Participant':<14}  " + "  ".join(f"{rm:>10}" for rm in ref_maps)
    print(header)
    print("-" * len(header))

    for participant, row in group.iterrows():
        flagged = any(row[rm] < MIN_IMAGES for rm in ref_maps)
        cells = "  ".join(f"{int(row[rm]):>10}" for rm in ref_maps)
        marker = "  <--" if flagged else ""
        print(f"{str(participant):<14}  {cells}{marker}")

    n_flagged = (group < MIN_IMAGES).any(axis=1).sum()
    print(f"\n{n_flagged}/{len(group)} participants below {MIN_IMAGES} entries in at least one reference map.")
