"""How many image entries does each participant contribute to the cross-phase NSS?

One table per awareness state, split by reference map (Intact / Scrambled), with
a "<--" marker on any participant below MIN_IMAGES. A quick way to spot who is thin
enough to be worth a closer look in DiagnoseParticipantCoverage.py.

Counts rows, i.e. scoring events (image x trial), so an image seen on two trials
counts twice. Both are valid data points.

Reads:  analysesresults/NSS_<mode>/NSS_CrossPhase_LongFormat.csv   (NSS.py)
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for the imports below
from Scripts.Analysis.NSS import NSSPaths
_P = NSSPaths.select()  # prompt or $MOONEY_SPLIT -> per-mode folder

CROSS_CSV  = _P["CROSS_CSV"]
from Settings import MIN_IMAGES_PER_PARTICIPANT as MIN_IMAGES  # flag anyone below this

# --- Load and count image entries per participant x awareness x reference map
df = pd.read_csv(CROSS_CSV)

counts = (
    df.groupby(["Awareness", "Participant", "ReferenceMap"])["Image"]
    .size()  # count rows (scoring events), so trial repeats count separately
    .unstack("ReferenceMap", fill_value=0)  # one column per reference map
    .sort_index()
)

ref_maps = list(counts.columns)  # e.g. ["Intact", "Scrambled"]

# --- Print one table per awareness state
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
