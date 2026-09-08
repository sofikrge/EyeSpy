"""Cross-phase NSS split-violin plot by awareness state. The main results figure.

    X-axis groups : Conscious Aware (PAS 2-3) | Unconscious Aware (PAS 2-3) | Unconscious Unaware (PAS 0)
    Split violin  : left half = Intact disambiguator reference, right half = Scrambled
    Overlaid dots : per-participant mean NSS, dodge-aligned with jitter off so
                    they form a single vertical line within each half-violin.

A whole-window figure with no Early/Late dimension, so it always reads the whole-mode
results (the halves comparison has its own plot, CrossNSSHalvesLinePlot.py). It takes the
trial set and blink mode from Settings.py and suffixes the output PNG to match, and it
follows the dataset switch: under EYESPY_DATASET=rep it reads analysesresults_rep/ and
writes to Figures_rep/ with a _rep suffix, so pilot and replication figures never mix.
The model EMM diamonds are hand-pasted from one specific lmer fit, so DRAW_EMMS is off
by default.

Reads:  analysesresults[_rep]/NSS_whole[_suffix]/NSS_CrossPhase_LongFormat.csv   (NSS.py)
Writes: Figures[_rep]/nss_analyses/NSS_CrossPhase_Violin_byAwareness*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIG ===
# Toggle which sessions to plot:
#   True  -> only the unconscious session (Unconscious Aware + Unconscious Unaware)
#   False -> all three groups (Conscious Aware + both unconscious groups)
UNCONSCIOUS_ONLY = True

# Trial set and blink mode (prompts, or the $TRIAL_SET / $BLINK_MODE env vars) pick
# which whole-mode results folder is read.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root, for the imports below
from Scripts.Analysis.NSS import NSSPaths
from Settings import _SUFFIX as DATASET_SUFFIX  # "" for the pilot, "_rep" under EYESPY_DATASET=rep
TRIAL_SET = NSSPaths.ask_trial_set()
BLINK_MODE = NSSPaths.ask_blink_mode()  # filter (original) / interp (PCHIP); picks the *_interp folder

# No Early/Late dimension here, so always the whole-mode results.
INPUT_FILE  = NSSPaths.paths_for("whole", TRIAL_SET, BLINK_MODE)["CROSS_CSV"]
# Replication figures go to Figures_rep/ and carry a _rep filename suffix, so a rep run
# can never overwrite (or be mistaken for) the pilot figure of the same mode.
OUTPUT_DIR  = Path(f"Figures{DATASET_SUFFIX}/nss_analyses") ; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_SUFFIX = NSSPaths.TRIAL_SET_SUFFIX[TRIAL_SET] + NSSPaths.BLINK_SUFFIX[BLINK_MODE] + DATASET_SUFFIX
OUTPUT_PLOT = OUTPUT_DIR / (
    f"NSS_CrossPhase_Violin_byAwareness_unconsciousOnly{_SUFFIX}.png"
    if UNCONSCIOUS_ONLY else
    f"NSS_CrossPhase_Violin_byAwareness{_SUFFIX}.png"
)

PALETTE = ['#edf8fb', '#b3cde3', '#648fff', '#785ef0']
REF_COLORS = {"Intact": PALETTE[3], "Scrambled": PALETTE[2]}

# Map (Session, Awareness) -> display label. The script prints the unique pairs it
# found at runtime, so edit these keys to match if the labels ever change.
GROUP_MAP = {
    ("C", "conscious_aware"):   "Conscious Aware\n(PAS 2-3)",
    ("U", "unconscious_aware"):   "Unconscious Aware\n(PAS 2-3)",
    ("U", "unconscious_unaware"): "Unconscious Unaware\n(PAS 0)",
}
GROUP_ORDER = ["Conscious Aware\n(PAS 2-3)", "Unconscious Aware\n(PAS 2-3)", "Unconscious Unaware\n(PAS 0)"]
if UNCONSCIOUS_ONLY:
    GROUP_ORDER = ["Unconscious Aware\n(PAS 2-3)", "Unconscious Unaware\n(PAS 0)"]


# Model EMMs, pasted in by hand from the fitted lmer. Drawn whenever DRAW_EMMS is True
# regardless of DATASET / TRIAL_SET / BLINK_MODE, so make sure they match the model you ran.
# Off by default: the values below are the pilot's all-trials model, so they are wrong for
# any other dataset or trial set. Set True again once you paste in matching EMMs.
DRAW_EMMS = False

# With image filtering
#EMMS = {
#    "Conscious Aware\n(PAS 2-3)":     {"Intact": 2.21, "Scrambled": 1.23},
#    "Unconscious Aware\n(PAS 2-3)":   {"Intact": 2.04, "Scrambled": 1.60},
#    "Unconscious Unaware\n(PAS 0)":   {"Intact": 2.10, "Scrambled": 2.31}
#}

# Without image filtering
EMMS = {
    "Conscious Aware\n(PAS 2-3)":     {"Intact": 2.21, "Scrambled": 1.23},
    "Unconscious Aware\n(PAS 2-3)":   {"Intact": 2.06, "Scrambled": 1.69},
    "Unconscious Unaware\n(PAS 0)":   {"Intact": 2.07, "Scrambled": 2.32}
}

# Write each participant's number next to their dot, to identify the individuals whose
# Intact-vs-Scrambled line runs against their group. Intact labels sit to the left of
# their dots and Scrambled ones to the right, so the text stays clear of the lines
# crossing between the two halves.
ANNOTATE_PARTICIPANTS = True

DOT_OFFSET = 0.17  # horizontal nudge so dots sit under each half-violin; tweak if misaligned
REF_OFFSET = {"Intact": -DOT_OFFSET, "Scrambled": DOT_OFFSET}
GROUP_POS  = {g: i for i, g in enumerate(GROUP_ORDER)}

def main():
    df = pd.read_csv(INPUT_FILE)

    print("Unique (Session, Awareness) pairs found in data:")
    print(df[["Session", "Awareness"]].drop_duplicates().to_string(index=False))

    # 1. Map to the 3 target groups; anything unmatched (e.g. C+Unaware, if present) is dropped
    df["Group"] = list(zip(df["Session"], df["Awareness"]))
    df["Group"] = df["Group"].map(GROUP_MAP)
    n_dropped = df["Group"].isna().sum()
    if n_dropped:
        print(f"Dropping {n_dropped} rows that didn't match GROUP_MAP (check Awareness labels above).")
    df = df.dropna(subset=["Group"])

    # Keep only the groups requested by the UNCONSCIOUS_ONLY toggle
    df = df[df["Group"].isin(GROUP_ORDER)]

    # 2. Collapse to one mean NSS per Participant x Group x ReferenceMap (across images/trials)
    df_agg = (
        df.groupby(["Participant", "Group", "ReferenceMap"], as_index=False)["NSS"]
          .mean()
    )

    df_agg["x_pos"] = df_agg["Group"].map(GROUP_POS) + df_agg["ReferenceMap"].map(REF_OFFSET)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.violinplot(
        data=df_agg, x="Group", y="NSS", hue="ReferenceMap",
        order=GROUP_ORDER, hue_order=["Intact", "Scrambled"],
        split=True, inner=None, cut=0, alpha=0.5,
        palette=REF_COLORS, linewidth=1, ax=ax
    )

    # Connect each participant's dots across the 3 awareness groups, within each reference map
    # Connect each participant's Intact and Scrambled dots WITHIN each awareness group
    for (participant, group), sub in df_agg.groupby(["Participant", "Group"]):
        sub = sub.sort_values("x_pos")
        # Only draw a line if the participant has both an Intact and a Scrambled score for this group
        if len(sub) == 2:
            ax.plot(sub["x_pos"], sub["NSS"], color="gray", alpha=0.35, linewidth=0.9, zorder=2)
    # Dots at fixed x_pos -> single vertical line per half-violin
    ax.scatter(df_agg["x_pos"], df_agg["NSS"], color="grey", 
               linewidth=0.5, s=20, alpha=0.8, zorder=3)

    if ANNOTATE_PARTICIPANTS:
        # Participants with near-identical means would print their numbers on top of each
        # other, so each half-violin's labels are walked bottom-up and pushed apart to a
        # minimum spacing. A thin leader line then ties a moved label back to its own dot.
        y_lo, y_hi = ax.get_ylim()
        min_gap = (y_hi - y_lo) * 0.022
        for _, col in df_agg.groupby(["Group", "ReferenceMap"]):
            col = col.sort_values("NSS")
            side = -1 if col["ReferenceMap"].iloc[0] == "Intact" else 1
            label_y = []
            for y in col["NSS"]:
                label_y.append(y if not label_y else max(y, label_y[-1] + min_gap))
            for row, y_lab in zip(col.itertuples(), label_y):
                ax.annotate(
                    str(row.Participant),
                    xy=(row.x_pos, row.NSS),                    # the dot itself
                    xytext=(row.x_pos + side * 0.04, y_lab),    # the decluttered label
                    ha="right" if side < 0 else "left", va="center",
                    fontsize=6.5, color="#333333", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4,
                                    shrinkA=0, shrinkB=2),
                )

    # Overlay the hand-pasted EMM diamonds whenever DRAW_EMMS is on (mode-independent).
    _draw_emms = DRAW_EMMS
    if not _draw_emms:
        print("Skipping EMM diamonds: DRAW_EMMS is False.")
    for group in (GROUP_ORDER if _draw_emms else []):
        for ref in ["Intact", "Scrambled"]:
            # Reconstruct the exact X position for this specific violin half
            x_pos = GROUP_POS[group] + REF_OFFSET[ref]

            # Grab the value from the dictionary
            if group in EMMS and ref in EMMS[group]:
                emm_val = EMMS[group][ref]
                
                # Draw the diamond
                ax.scatter(
                    x=x_pos, 
                    y=emm_val, 
                    color="white", 
                    edgecolors="black", 
                    linewidth=1.2,
                    marker="D", # 'D' for Diamond
                    s=70,       # Size of the diamond
                    zorder=5    # Ensures it sits on top of all other dots and lines
                )

    # --- NEW: ADD MEAN SQUARES ---
    # Calculate the overall mean for each group + reference map combo
    means_df = df_agg.groupby(["Group", "ReferenceMap", "x_pos"], as_index=False)["NSS"].mean()
    
    # Plot the means as black squares
    ax.scatter(
        x=means_df["x_pos"], 
        y=means_df["NSS"], 
        color="#222222",      # Very dark grey/black
        edgecolors="white",   # White border to make it pop against the other dots
        linewidth=1.2,
        marker="s",           # "s" stands for square
        s=60,                 # Slightly larger than the raw data dots
        zorder=4              # Ensures they sit on top of the dots and lines
    )
    # -----------------------------

    # Drop the duplicate legend entries created by stripplot, keep only the violin's
    handles, labels = ax.get_legend_handles_labels()
    # Place the legend outside the axes (right side) so it never overlaps the data points.
    ax.legend(title="Disambiguator Type", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False)

    ax.set_xlabel("")
    ax.set_ylabel("NSS")
    ax.set_title("Cross-phase NSS by Awareness State")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()