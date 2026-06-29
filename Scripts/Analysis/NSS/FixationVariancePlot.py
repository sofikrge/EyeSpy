# FixationVariancePlot.py

"""
Exploration / spread plots by Awareness State and Image Type
============================================================
Same figure setup as CrossNSSViolinPlot.py, but the y-axis describes where/how
spread-out participants fixate, NOT cross-phase NSS. Two figures:

  1. Distance from centre  - "do they explore the whole stimulus or only its centre?"
  2. Spatial variance      - variance of fixation locations; low variance can have
                             strange effects on correlations (professor's main ask).

Both use the fixation locations already in the parquet ((0, 0) = image centre):
  distance = hypot(x, y)          per fixation, then averaged per participant
  variance = Var(x) + Var(y)      across a participant's fixations of that type

Input : data/NSS_all_fixations_clean.parquet  (one row per fixation, NSSExporter.py)
Output: Figures/nss_analyses/FixationEccentricity_byAwareness.png
        Figures/nss_analyses/FixationSpatialVariance_byAwareness.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_FILE = Path("data/NSS_all_fixations_clean.parquet")
OUTPUT_DIR = Path("Figures/nss_analyses"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_MAP = {
    "conscious_aware":    "Conscious Aware\n(PAS 2-3)",
    "unconscious_aware":   "Unconscious Aware\n(PAS 2-3)",
    "unconscious_unaware": "Unconscious Unaware\n(PAS 0)",
}
TYPE_MAP = {
    "mooney_post_intact": "Post-Intact Mooney",
    "disamb_intact":      "Intact Disambiguator",
    "disamb_not_intact":  "Scrambled Disambiguator",
}
GROUP_ORDER = list(GROUP_MAP.values())
TYPE_ORDER  = list(TYPE_MAP.values())
PALETTE = ['#648fff', '#785ef0', '#dc267f']
TYPE_COLORS = dict(zip(TYPE_ORDER, PALETTE))
HUE_OFFSET = {t: (i - 1) * (0.8 / 3) for i, t in enumerate(TYPE_ORDER)}
GROUP_POS  = {g: i for i, g in enumerate(GROUP_ORDER)}


def plot(agg, col, ylabel, title, out_name):
    agg = agg.copy()
    agg["x_pos"] = agg["Group"].map(GROUP_POS) + agg["Type"].map(HUE_OFFSET)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(data=agg, x="Group", y=col, hue="Type",
                   order=GROUP_ORDER, hue_order=TYPE_ORDER,
                   inner=None, cut=0, alpha=0.5,
                   palette=TYPE_COLORS, linewidth=1, ax=ax)

    for (_pid, _grp), sub in agg.groupby(["participant", "Group"]):
        sub = sub.sort_values("x_pos")
        if len(sub) >= 2:
            ax.plot(sub["x_pos"], sub[col], color="gray", alpha=0.30, linewidth=0.8, zorder=2)
    ax.scatter(agg["x_pos"], agg[col], color="grey", linewidth=0.5, s=18, alpha=0.8, zorder=3)

    means = agg.groupby(["Group", "Type", "x_pos"], observed=True)[col].mean().reset_index()
    ax.scatter(means["x_pos"], means[col], color="#222222", edgecolors="white",
               linewidth=1.2, marker="s", s=60, zorder=4)

    ax.legend(title="Image Type", loc="upper right", frameon=False)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / out_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    fix = pd.read_parquet(INPUT_FILE)
    fix = fix[fix["image_type"].isin(TYPE_MAP)].copy()
    fix["Group"] = fix["awareness"].map(GROUP_MAP)
    fix["Type"]  = fix["image_type"].map(TYPE_MAP)
    fix = fix.dropna(subset=["Group", "Type"])
    fix["dist"] = np.hypot(fix["x_deg_centered"], fix["y_deg"])

    keys = ["participant", "Group", "Type"]

    # 1. Distance from centre: per-fixation distance averaged per participant
    ecc = fix.groupby(keys, observed=True)["dist"].mean().reset_index()
    plot(ecc, "dist", "Mean distance from centre  (deg)",
         "Fixation exploration by Awareness State and Image Type",
         "FixationEccentricity_byAwareness.png")

    # 2. Spatial variance: Var(x) + Var(y) across a participant's fixations
    var = (fix.groupby(keys, observed=True)
              .apply(lambda g: g["x_deg_centered"].var() + g["y_deg"].var())
              .rename("var").reset_index())
    plot(var, "var", "Spatial variance of fixations  (deg²)",
         "Fixation spatial variance by Awareness State and Image Type",
         "FixationSpatialVariance_byAwareness.png")


if __name__ == "__main__":
    main()
