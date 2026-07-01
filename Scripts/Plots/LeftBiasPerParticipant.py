"""
LeftBiasPerParticipant.py
-------------------------
Is the left-side fixation bias (seen in scrambled images + unaware Mooneys)
consistent across participants, or driven by a few?

Computes the SAME "Left %" diagnostic as FixationDensityPlot.py, but PER
PARTICIPANT instead of pooled across everyone. The grid mirrors
FixationDensityPlot.py exactly so the two figures line up cell-for-cell:

    Rows    - awareness state: Conscious Aware / Unconscious Aware / Unconscious Unaware
    Columns - image type:      Post-Intact Mooney / Intact Disambiguator / Scrambled Disambiguator

Filtering matches FixationDensityPlot.py / NSS.py:
  - Mooney column is awareness-split (filter by awareness group).
  - Disambiguator columns are SESSION-split, not awareness-split, so the
    Unconscious-Aware and Unconscious-Unaware rows show the SAME session-U data.

Each panel shows one dot per participant (dot size = fixation count) with the
50% no-bias line and the per-participant mean. A sorted table is printed per cell.

Reading the result:
  - If almost every participant sits above 50%  -> consistent, real bias.
  - If most cluster around 50% and a few extreme participants pull the pooled
    mean up -> the effect is driven by those few (check n per participant too).

Run from the project root:
    python3 Scripts/Analysis/NSSControlAnalyses/LeftBiasPerParticipant.py

Requires: data/NSS_all_fixations_clean.parquet  (produced by NSSExporter.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

FIX_FILE    = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR  = Path("Figures/nss_separated_analyses")
OUTPUT_FILE = OUTPUT_DIR / "LeftBiasPerParticipant.png"

IMAGE_HEIGHT = 600    # eye-tracking canvas height in pixels - must match NSS.py
IMAGE_WIDTH  = 800    # eye-tracking canvas width  in pixels - must match NSS.py
MASK_PPD     = 48.55  # pixels per visual degree - must match NSS.py

MIN_FIX_PER_PARTICIPANT = 20   # ignore participants with too little data in a cell

# =============================================================================
# GRID DEFINITION - identical to FixationDensityPlot.py
# =============================================================================

AWARENESS_ROWS = [
    ("conscious_aware",     "Conscious Aware\n(PAS 2-3)"),
    ("unconscious_aware",   "Unconscious Aware\n(PAS 2-3)"),
    ("unconscious_unaware", "Unconscious Unaware\n(PAS 0)"),
]

IMAGE_TYPE_COLS = [
    ("mooney_post_intact", "Post-Intact Mooney"),
    ("disamb_intact",      "Intact Disambiguator"),
    ("disamb_not_intact",  "Scrambled Disambiguator"),
]

# Disambiguator maps are session-split (not awareness-split) - same as NSS.py.
AWARENESS_TO_SESSION = {
    "conscious_aware":     "C",
    "unconscious_aware":   "U",
    "unconscious_unaware": "U",
}
SESSION_LABEL = {"C": "Session C", "U": "Session U"}

# =============================================================================
# COORDINATE HELPERS - exact mirror of NSS.py / FixationDensityPlot.py
# =============================================================================

def _round_half_away_from_zero(x):
    x = np.asarray(x, dtype=float)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)


def _deg_to_image_pixels(x_deg, y_deg):
    """Visual degrees -> 1-based (row, col) pixel coords, same as NSS.py."""
    w_1b = _round_half_away_from_zero((IMAGE_WIDTH  / 2.0) + np.asarray(x_deg, float) * MASK_PPD)
    h_1b = _round_half_away_from_zero((IMAGE_HEIGHT / 2.0) + np.asarray(y_deg, float) * MASK_PPD)
    return h_1b, w_1b


def left_pct(df):
    """Fraction (%) of in-bounds fixations in the left half + the in-bounds count."""
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    xdeg = pd.to_numeric(df["x_deg_centered"], errors="coerce").to_numpy()
    ydeg = pd.to_numeric(df["y_deg"],          errors="coerce").to_numpy()
    ok   = np.isfinite(xdeg) & np.isfinite(ydeg)
    if not ok.any():
        return np.nan, 0
    h1, w1 = _deg_to_image_pixels(xdeg[ok], ydeg[ok])
    inside = (h1 >= 1) & (h1 <= H) & (w1 >= 1) & (w1 <= W)
    n = int(inside.sum())
    if n == 0:
        return np.nan, 0
    return 100.0 * (w1[inside] <= W / 2).sum() / n, n


def per_participant_table(subset):
    """One row per participant with enough data: participant, left_pct, n."""
    rows = []
    for pid, df_p in subset.groupby("participant", dropna=False):
        pct, n = left_pct(df_p)
        if n >= MIN_FIX_PER_PARTICIPANT:
            rows.append((str(pid), pct, n))
    return (pd.DataFrame(rows, columns=["participant", "left_pct", "n"])
              .sort_values("left_pct").reset_index(drop=True))


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fix = pd.read_parquet(FIX_FILE)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharey=True)

    for row_idx, (awareness_key, awareness_label) in enumerate(AWARENESS_ROWS):
        session_key = AWARENESS_TO_SESSION[awareness_key]

        for col_idx, (img_type_key, img_type_label) in enumerate(IMAGE_TYPE_COLS):
            ax = axes[row_idx, col_idx]
            is_mooney = img_type_key == "mooney_post_intact"

            if is_mooney:
                subset = fix[(fix["awareness"]  == awareness_key) &
                             (fix["image_type"] == img_type_key)]
                cell_desc = f"{awareness_key} x {img_type_key}"
            else:
                # Disambiguator: session-split, not awareness-split (matches NSS.py)
                subset = fix[(fix["session"]    == session_key) &
                             (fix["image_type"] == img_type_key)]
                cell_desc = f"{SESSION_LABEL[session_key]} x {img_type_key}"

            tbl    = per_participant_table(subset)
            pooled, pooled_n = left_pct(subset)
            n_part   = len(tbl)
            n_biased = int((tbl["left_pct"] > 50).sum()) if n_part else 0

            print("=" * 64)
            print(cell_desc)
            print("=" * 64)
            print(f"  Pooled Left%        : {pooled:5.1f}%   (n = {pooled_n:,} fixations)"
                  if pooled_n else "  No data")
            if n_part:
                print(f"  Per-participant mean: {tbl['left_pct'].mean():5.1f}%")
                print(f"  Per-participant med : {tbl['left_pct'].median():5.1f}%")
                print(f"  Left-biased (>50%)  : {n_biased}/{n_part} participants "
                      f"({100*n_biased/n_part:.0f}%)")
                print(tbl.to_string(index=False,
                                    formatters={"left_pct": lambda v: f"{v:5.1f}"}))
            print()

            # --- strip plot ---
            if n_part:
                y = tbl["left_pct"].to_numpy()
                # Fresh per-panel RNG so identical data renders identically
                # (the two unconscious disamb panels are the same session-U data).
                x = np.random.default_rng(0).normal(0, 0.04, size=len(y))
                ax.scatter(x, y, s=np.clip(tbl["n"] / 5, 10, 300),
                           alpha=0.6, edgecolor="k", linewidth=0.4)
                ax.axhline(tbl["left_pct"].mean(), color="blue", linewidth=1,
                           label=f"mean {tbl['left_pct'].mean():.1f}%")
                ax.text(0.5, 0.02,
                        f"{n_biased}/{n_part} left-biased  ·  pooled {pooled:.1f}%",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.7, linewidth=0))
                ax.legend(fontsize=8, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="#666")

            if is_mooney:
                ax.text(0.5, 0.93, "split by AWARENESS",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.5, fontweight="bold", color="#0050b0",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.8, linewidth=0))

            ax.axhline(50, color="red", linestyle="--", linewidth=1)
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(0, 100)
            ax.set_xticks([])

            if not is_mooney:
                # Disamb panels are SESSION-split, not awareness-split: stamp the
                # session prominently so the awareness row label isn't misread.
                # (Both unconscious rows therefore show identical session-U data.)
                ax.text(0.5, 0.93,
                        f"split by SESSION: {SESSION_LABEL[session_key]}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.5, fontweight="bold", color="#b00020",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.8, linewidth=0))
            if row_idx == 0:
                ax.set_title(img_type_label, fontsize=13, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(awareness_label + "\n\n% fixations on LEFT",
                              fontsize=10, fontweight="bold")

    fig.suptitle("Left-side fixation bias per participant  "
                 "(dot = participant, size = fixation count, red = 50% no-bias)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
