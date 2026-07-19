"""
Double group-mean centre the within-phase NSS ("gaze typicality") covariate.

Reads the cross-phase long-format results, removes the between-participant and
between-image variance from Within-NSS-Typicality, and writes a new column
`GazeTypicalityCentred` alongside the untouched original so the two can be
compared in the mixed model.

    x_within = x - x_participant - x_image + x_grand   (Guo et al., 2024)

The original `Within-NSS-Typicality` column is left unchanged.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for nss_paths
import nss_paths

_P = nss_paths.select()  # prompt or $MOONEY_SPLIT -> per-mode folder
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
