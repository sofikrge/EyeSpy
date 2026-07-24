"""Where should interpolation be capped?  (blink-interpolation diagnostic 1 of 3)

Which blinks are real blinks, worth interpolating, and which are long track losses that
must be left as gaps? Justifies Settings.MAX_BLINK_INTERP_MS.

It reads every raw .asc, takes each EyeLink blink (EBLINK marker) duration, and reports
the pooled distribution + the fraction of blinks above candidate caps + a histogram. The
distribution is bimodal: a physiological-blink mode (~50-150 ms) and a long track-loss
tail; the chosen cap should sit in the trough between them.

Reads the raw .asc files directly, so it needs no pipeline run. From the project root:
    python3 Scripts/Preprocessing/Diagnostics/BlinkDurationDistribution.py

Writes (under DataQualityChecks/blink_interpolation/):
    blink_duration_distribution.csv   - histogram bins + summary percentiles
    blink_duration_histogram.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for imports below

import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Settings as settings
from Scripts.Preprocessing.FileParsing import parse_blink_intervals

OUT = Path("DataQualityChecks/blink_interpolation"); OUT.mkdir(parents=True, exist_ok=True)
CAP = settings.MAX_BLINK_INTERP_MS  # the interpolation cap this diagnostic is checking

def main():
    files = sorted(glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.asc")))
    durations = []
    for f in files:
        durations += [off - on for on, off in parse_blink_intervals(f)]  # EBLINK onset/offset are ms @1000Hz
    d = np.array(durations)
    print(f"Files: {len(files)}   pooled blinks: {len(d)}")

    # --- summary percentiles ---
    pct = {p: float(np.percentile(d, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
    print(f"median={pct[50]:.0f} ms   95th={pct[95]:.0f} ms   max={d.max()} ms")
    for cap in (200, 300, 400, 500, 750, 1000, 2000):
        n = int((d > cap).sum())
        print(f"  blinks > {cap:>4} ms: {n:5d}  ({100*n/len(d):5.2f}%)  <- cap={CAP}" if cap == CAP else
              f"  blinks > {cap:>4} ms: {n:5d}  ({100*n/len(d):5.2f}%)")

    # --- histogram (finite bins + overflow) ---
    edges = [0,50,100,150,200,250,300,350,400,450,500,750,1000,2000,10**9]
    labels = ['0-50','50-100','100-150','150-200','200-250','250-300','300-350','350-400',
              '400-450','450-500','500-750','750-1000','1000-2000','2000+']
    counts, _ = np.histogram(d, bins=edges)

    # --- save report CSV ---
    rows = [{"bin_ms": lab, "count": int(c)} for lab, c in zip(labels, counts)]
    rows += [{"bin_ms": f"pct_{p}", "count": round(v, 1)} for p, v in pct.items()]
    rows.append({"bin_ms": "n_blinks", "count": len(d)})
    rows.append({"bin_ms": "cap_ms", "count": CAP})
    pd.DataFrame(rows).to_csv(OUT / "blink_duration_distribution.csv", index=False)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(labels)), counts, color="#8c96c6", edgecolor="white")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("blink count"); ax.set_xlabel("blink duration (ms)")
    # mark where the cap falls (between the 450-500 and 500-750 bins -> index 10 boundary)
    ax.axvline(9.5, color="#c1121f", ls="--", lw=1.5, label=f"interpolation cap = {CAP} ms")
    ax.set_title(f"Blink-duration distribution (n={len(d)}): cap sits in the trough before the track-loss tail")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "blink_duration_histogram.png", dpi=130)
    print(f"Saved -> {OUT/'blink_duration_histogram.png'} and blink_duration_distribution.csv")

if __name__ == "__main__":
    main()
