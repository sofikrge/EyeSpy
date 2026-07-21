# BlinkContaminationProfile.py
"""
Diagnostic 2/3 for the blink-INTERPOLATION preprocessing option.

Question it answers: "how big should the peri-blink margin be?" i.e. how far before/after
a blink is the gaze signal contaminated (so those samples must not anchor the interpolation).
Justifies Settings.BLINK_MARGIN_MS.

Method: a blink-locked average. Every blink in every raw .asc is aligned to its onset (and
separately to its offset), and pupil size + gaze speed are averaged across all blinks at each
time offset. Contamination shows up as a departure from the far baseline. In this dataset the
pupil/velocity depart ~60 ms before onset and take ~150 ms to recover after offset, so a
symmetric margin must cover ~150 ms; 51 ms is too short, 200 ms clears it.

Pupil during a blink is logged as 0 by EyeLink -> treated as missing here; position is '.'
(-> NaN). Only windows that stay within one contiguous recording segment are averaged.

Outputs (under DataQualityChecks/blink_interpolation/):
    blink_contamination_profile.png    - 2x2: pupil & gaze-speed, onset- and offset-locked
    blink_contamination_report.txt     - where the signal departs / recovers
Run from the project root:  python3 Scripts/Preprocessing/Diagnostics/BlinkContaminationProfile.py
(Reads all raw samples; takes ~2 min.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Settings as settings

OUT = Path("DataQualityChecks/blink_interpolation"); OUT.mkdir(parents=True, exist_ok=True)
WIN = 350          # ms of context on each side of a blink edge
MARGIN = settings.BLINK_MARGIN_MS   # the margin this diagnostic is checking
OLD_BUFFER = 51    # the primary filter method's buffer, for comparison

def _fnum(v):
    return np.nan if v == "." else float(v)

def _parse(path):
    """Return (t, x, y, pupil, blinks) for one .asc; missing position/pupil -> NaN."""
    ts=[]; xs=[]; ys=[]; ps=[]; blinks=[]
    with open(path, errors="ignore") as fh:
        for line in fh:
            c = line[0] if line else ""
            if c.isdigit():
                p = line.split()
                if len(p) >= 4:
                    ts.append(int(p[0])); xs.append(_fnum(p[1])); ys.append(_fnum(p[2])); ps.append(_fnum(p[3]))
            elif c == "E" and line.startswith("EBLINK"):
                p = line.split(); blinks.append((int(p[2]), int(p[3])))
    return np.array(ts, np.int64), np.array(xs), np.array(ys), np.array(ps), blinks

def main():
    files = sorted(glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.asc")))
    off = np.arange(-WIN, WIN + 1)
    def acc(): return {"ps": np.zeros(off.size), "pn": np.zeros(off.size),
                       "vs": np.zeros(off.size), "vn": np.zeros(off.size)}
    ON, OFFA = acc(), acc()   # onset-aligned, offset-aligned

    n_used = 0
    for fi, f in enumerate(files):
        t, x, y, pupil, blinks = _parse(f)
        if t.size == 0:
            continue
        vel = np.hypot(np.gradient(x), np.gradient(y))   # px/ms
        pupil = np.where(pupil > 0, pupil, np.nan)        # 0 pupil = blink -> missing
        for b_on, b_off in blinks:
            for edge, D in ((b_on, ON), (b_off, OFFA)):
                i = np.searchsorted(t, edge)
                lo, hi = i - WIN, i + WIN + 1
                if lo < 0 or hi > t.size:
                    continue
                seg = t[lo:hi]
                if seg[-1] - seg[0] != (hi - lo - 1):     # contiguous 1 ms window only
                    continue
                pv, vv = pupil[lo:hi], vel[lo:hi]
                mp = ~np.isnan(pv); D["ps"][mp] += pv[mp]; D["pn"][mp] += 1
                mv = ~np.isnan(vv); D["vs"][mv] += vv[mv]; D["vn"][mv] += 1
            n_used += 1
        print(f"[{fi+1}/{len(files)}] {os.path.basename(f)}  blinks={len(blinks)}", flush=True)

    on_p = ON["ps"]/np.maximum(ON["pn"],1); on_v = ON["vs"]/np.maximum(ON["vn"],1)
    off_p = OFFA["ps"]/np.maximum(OFFA["pn"],1); off_v = OFFA["vs"]/np.maximum(OFFA["vn"],1)

    # --- report: departure (onset side) / recovery (offset side) vs baseline ---
    def cross(sig, base_mask, side, rising):
        mu, sd = np.nanmean(sig[base_mask]), np.nanstd(sig[base_mask])
        thr = mu + 3*sd if rising else mu - 3*sd
        within = [o for o in off if (0 > o > -250 if side == "on" else 0 < o < 250)]
        hits = [o for o in within if (sig[off == o][0] > thr if rising else sig[off == o][0] < thr)]
        return (min(hits) if hits else None) if side == "on" else (max(hits) if hits else None)

    far_on, far_off = (off <= -250), (off >= 250)
    lines = [
        f"blinks contributing: {n_used}",
        f"pupil starts dropping ~{cross(on_p, far_on, 'on', rising=False)} ms before blink onset",
        f"pupil recovers by      ~{cross(off_p, far_off, 'off', rising=False)} ms after blink offset",
        f"gaze speed departs     ~{cross(on_v, far_on, 'on', rising=True)} ms before onset",
        f"gaze speed settles by  ~{cross(off_v, far_off, 'off', rising=True)} ms after offset",
        f"=> chosen margin BLINK_MARGIN_MS = {MARGIN} ms (must cover the slower, offset side)",
    ]
    (OUT / "blink_contamination_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- plot: 2 rows (pupil, speed) x 2 cols (onset, offset) ---
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    panels = [(axs[0,0], on_p, np.nanmean(on_p[far_on]), "Pupil - aligned to ONSET", "pupil area (a.u.)", "#2e5cb8"),
              (axs[0,1], off_p, np.nanmean(off_p[far_off]), "Pupil - aligned to OFFSET", "pupil area (a.u.)", "#2e5cb8"),
              (axs[1,0], on_v, np.nanmean(on_v[far_on]), "Gaze speed - aligned to ONSET", "gaze speed (px/ms)", "#c1121f"),
              (axs[1,1], off_v, np.nanmean(off_v[far_off]), "Gaze speed - aligned to OFFSET", "gaze speed (px/ms)", "#c1121f")]
    for ax, sig, base, ttl, ylab, col in panels:
        left = ax in (axs[0,0], axs[1,0])
        ax.plot(off, sig, color=col)
        ax.axhline(base, color="grey", ls=":", lw=1)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(-MARGIN if left else MARGIN, color="#e08a00", ls="--", lw=1.3)
        ax.axvline(-OLD_BUFFER if left else OLD_BUFFER, color="#3b7a57", ls="--", lw=1.3)
        ax.set_title(ttl, fontsize=10); ax.set_ylabel(ylab); ax.grid(alpha=.25)
    for ax in axs[1]:
        ax.set_xlabel("time from blink edge (ms)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],color="black"), Line2D([],[],color="#e08a00",ls="--"),
                Line2D([],[],color="#3b7a57",ls="--"), Line2D([],[],color="grey",ls=":")],
               ["blink edge (0)", f"+/-{MARGIN} ms margin (used)", f"+/-{OLD_BUFFER} ms (filter buffer)", "far baseline"],
               ncol=4, loc="upper center", fontsize=9)
    fig.suptitle(f"Blink-locked averages over {n_used} blinks - peri-blink contamination", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT / "blink_contamination_profile.png", dpi=130)
    print(f"Saved -> {OUT/'blink_contamination_profile.png'} and blink_contamination_report.txt")

if __name__ == "__main__":
    main()
