"""Double group-mean centre the within-phase NSS ("gaze typicality") covariate.

The cross-phase data is cross-classified: every row sits in both a participant and
an image, so centring on one grouping alone leaves the other's variance in the
covariate. Subtracting both group means isolates the within-cell component:

    x_within = x - x_participant - x_image + x_grand   (Guo et al., 2024)

The result goes in a new `GazeTypicalityCentred` column; the original
`Within-NSS-Typicality` is left untouched so raw and centred models can be
compared. In Jamovi, use the centred column with covariate scaling set to None.

Reads:  analysesresults/NSS_<mode>/NSS_CrossPhase_LongFormat.csv          (NSS.py)
Writes: analysesresults/NSS_<mode>/NSS_CrossPhase_LongFormat_centred.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for the imports below
from Scripts.Analysis.NSS import NSSPaths

_P = NSSPaths.select()  # prompt or $MOONEY_SPLIT -> per-mode folder
IN_PATH = _P["CROSS_CSV"]
OUT_PATH = _P["CROSS_CENTRED_CSV"]
COL = "Within-NSS-Typicality"

df = pd.read_csv(IN_PATH)

# Compute the group means on unique participant x image x session x trial cells.
# Typicality is per viewing (constant within a trial), so dedup on Trial too: this
# collapses the duplicated Intact/Scrambled reference rows (and, in halves mode, the
# Early/Late rows) that share a value, while keeping each repeat viewing's distinct
# typicality as its own observation.
cells = df.dropna(subset=[COL]).drop_duplicates(["Participant", "Image", "Session", "Trial"])

grand = cells[COL].mean()
ppt_mean = cells.groupby("Participant")[COL].mean()
img_mean = cells.groupby("Image")[COL].mean()

df["GazeTypicalityCentred"] = (
    df[COL]
    - df["Participant"].map(ppt_mean)
    - df["Image"].map(img_mean)
    + grand
)

df.to_csv(OUT_PATH, index=False)

print(f"Wrote {OUT_PATH}")
print(f"  rows: {len(df)}   cells used for means: {len(cells)}")
print(f"  grand mean: {grand:.4f}")
print(df[[COL, "GazeTypicalityCentred"]].describe().round(4))
