# NSS_CrossPhase_ViolinByAwareness.py

"""
Cross-Phase NSS Split-Violin Plot by Awareness State
======================================================
Reads the participant-trial-level long format produced by the awareness-aware
cross-phase NSS export (NSS_CrossPhase_LongFormat.csv, columns: Participant,
Image, Session, Awareness, Trial, Experiment_Half, ReferenceMap, NSS).

Plot:
    X-axis groups : Conscious Aware (PAS 2-3) | Unconscious Aware (PAS 2-3) | Unconscious Unaware (PAS 0)
    Split violin  : left half = Intact disambiguator reference, right half = Scrambled
    Overlaid dots : per-participant mean NSS, dodge-aligned with jitter off so
                    they form a single vertical line within each half-violin.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIG ===
INPUT_FILE  = Path("analysesresults/NSS/NSS_CrossPhase_LongFormat.csv")
OUTPUT_DIR  = Path("Figures/nss_analyses") ; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT = OUTPUT_DIR / "NSS_CrossPhase_Violin_byAwareness.png"

PALETTE = ['#edf8fb', '#b3cde3', '#648fff', '#785ef0']
REF_COLORS = {"Intact": PALETTE[3], "Scrambled": PALETTE[2]}

# Map (Session, Awareness) -> display label.
# If your "Awareness" column uses different strings than "Aware"/"Unaware"
# (case, spelling, etc.), the script prints the unique pairs it found at
# runtime — edit the keys below to match those exactly.
GROUP_MAP = {
    ("C", "conscious_aware"):   "Conscious Aware\n(PAS 2-3)",
    ("U", "unconscious_aware"):   "Unconscious Aware\n(PAS 2-3)",
    ("U", "unconscious_unaware"): "Unconscious Unaware\n(PAS 0)",
}
GROUP_ORDER = ["Conscious Aware\n(PAS 2-3)", "Unconscious Aware\n(PAS 2-3)", "Unconscious Unaware\n(PAS 0)"]


EMMS = {
    "Conscious Aware\n(PAS 2-3)":     {"Intact": 2.21, "Scrambled": 1.23},
    "Unconscious Aware\n(PAS 2-3)":   {"Intact": 1.81, "Scrambled": 1.22},
    "Unconscious Unaware\n(PAS 0)":   {"Intact": 1.94, "Scrambled": 2.20}
}

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

    for group in GROUP_ORDER:
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
    ax.legend(title="Reference Map", loc="upper right", frameon=False)

    ax.set_xlabel("")
    ax.set_ylabel("NSS")
    ax.set_title("Cross-phase NSS by Awareness State")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()