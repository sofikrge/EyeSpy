"""What does the margin cost us?  (blink-interpolation diagnostic 3 of 3)

How much of the analyzed gaze would be reconstructed rather than measured if we
interpolate? Confirms the margin is affordable in the phases that actually matter,
above all the Mooney phase, which carries the DV.

For each raw .asc it reconstructs the analyzed phase windows (disambiguation and Mooney,
from the MSG markers via parse_trials_from_asc) and the blinks (EBLINK). For candidate
margins it reports the % of analyzed time that falls inside a blink +/- margin (i.e. would
be PCHIP-filled), split by phase. Only blinks up to Settings.MAX_BLINK_INTERP_MS count
(longer = track loss, left as gaps). Overlapping fill windows are merged so nothing is
double counted.

Reads the raw .asc files directly, so it needs no pipeline run. From the project root:
    python3 Scripts/Preprocessing/Diagnostics/BlinkInterpolationFraction.py

Writes (under DataQualityChecks/blink_interpolation/):
    blink_interpolation_fraction.csv   - analyzed minutes + reconstructed % per phase per margin
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import glob
import os
import pandas as pd

import Settings as settings
from Scripts.Preprocessing.FileParsing import parse_trials_from_asc, parse_blink_intervals

OUT = Path("DataQualityChecks/blink_interpolation"); OUT.mkdir(parents=True, exist_ok=True)
CAP = settings.MAX_BLINK_INTERP_MS
MARGINS = [51, 150, settings.BLINK_MARGIN_MS]   # 51 = old filter buffer; last = chosen margin

def _merge(intervals):
    """Merge overlapping [a,b] intervals so overlap is never double counted."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def _overlap(fills, windows):
    """Total ms of merged `fills` that lie inside any (start, end) in `windows`."""
    total = 0
    for ws, we in windows:
        for fa, fb in fills:
            o = min(fb, we) - max(fa, ws)
            if o > 0:
                total += o
    return total

def main():
    files = sorted(glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.asc")))
    phases = ("disamb", "mooney")
    agg = {ph: {"analyzed": 0, "rawblink": 0, **{m: 0 for m in MARGINS}} for ph in phases}

    for f in files:
        trials = parse_trials_from_asc(f, labels=settings.TRIAL_LABELS, patterns=settings.ASC_PATTERNS)
        if trials.is_empty():
            continue
        short = [(on, off) for on, off in parse_blink_intervals(f) if off - on <= CAP]

        wins = {"disamb": [], "mooney": []}
        for row in trials.to_dicts():
            if row.get("disambig_start") and row.get("disambig_end"):
                wins["disamb"].append((row["disambig_start"], row["disambig_end"]))
            if row.get("mooney_start") and row.get("mooney_end"):
                wins["mooney"].append((row["mooney_start"], row["mooney_end"]))

        for ph in phases:
            W = wins[ph]
            agg[ph]["analyzed"] += sum(we - ws for ws, we in W)
            agg[ph]["rawblink"] += _overlap(_merge([list(b) for b in short]), W)
            for m in MARGINS:
                agg[ph][m] += _overlap(_merge([[on - m, off + m] for on, off in short]), W)

    # --- build report ---
    def pct(ph, key):
        a = agg[ph]["analyzed"]
        return round(100 * agg[ph][key] / a, 1) if a else 0.0

    rows = []
    for ph in phases:
        rows.append({"phase": ph, "analyzed_min": round(agg[ph]["analyzed"]/60000, 1),
                     "raw_blink_pct": pct(ph, "rawblink"),
                     **{f"margin{m}_pct": pct(ph, m) for m in MARGINS}})
    tot_a = sum(agg[ph]["analyzed"] for ph in phases)
    both = {"phase": "BOTH", "analyzed_min": round(tot_a/60000, 1),
            "raw_blink_pct": round(100*sum(agg[ph]["rawblink"] for ph in phases)/tot_a, 1)}
    for m in MARGINS:
        both[f"margin{m}_pct"] = round(100*sum(agg[ph][m] for ph in phases)/tot_a, 1)
    rows.append(both)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "blink_interpolation_fraction.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {OUT/'blink_interpolation_fraction.csv'}  (chosen margin = {settings.BLINK_MARGIN_MS} ms)")

if __name__ == "__main__":
    main()
