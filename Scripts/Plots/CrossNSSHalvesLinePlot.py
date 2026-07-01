"""Cross-phase NSS plot of model-based estimated marginal means (EMMs).

Single panel. For each awareness state, two points side by side: First half
(left) and Second half (right), connected by a short line. Reference maps are
coloured separately: Intact (blue) vs Scrambled (green); second-half dots use a
darker shade. Error bars are the 95% CIs from the linear mixed model.

EMMs are pasted in below from the fitted lmer model:
    NSS ~ Awareness * ReferenceMap * Experiment_Half
          + (1 + ReferenceMap | Participant) + (1 + ReferenceMap | Image)
(Satterthwaite df, Wald CIs).

Output: Figures/nss_separated_analyses/CrossNSSHalvesLinePlot.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def darken(color, factor=0.55):
    """Return a darker shade of `color` (factor < 1 = darker)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

# === CONFIG ===
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "Figures" / "nss_separated_analyses" / "CrossNSSHalvesLinePlot.png"

AWARENESS_ORDER = ["conscious_aware", "unconscious_aware", "unconscious_unaware"]
HALVES = ["First_Half", "Second_Half"]
COLORS = {"Intact": "tab:blue", "Scrambled": "tab:green"}
# Layout within each awareness state: Intact pair dodged left, Scrambled right,
# so their CIs never overlap; HALF_OFFSET separates the two halves within a pair.
REF_DODGE = {"Intact": -0.10, "Scrambled": 0.10}
HALF_OFFSET = 0.05

# Model EMMs: (Awareness, ReferenceMap, Half) -> (mean, ci_lower, ci_upper).
# Pasted from Jamovi GAMLj "Estimated Marginal Means" (95% Wald CIs).
EMMS = {
    ("conscious_aware",     "Intact",    "First_Half"):  (2.35, 1.97, 2.72),
    ("conscious_aware",     "Intact",    "Second_Half"): (2.12, 1.74, 2.49),
    ("conscious_aware",     "Scrambled", "First_Half"):  (1.23, 0.87, 1.60),
    ("conscious_aware",     "Scrambled", "Second_Half"): (1.17, 0.80, 1.54),
    ("unconscious_aware",   "Intact",    "First_Half"):  (1.69, 1.30, 2.08),
    ("unconscious_aware",   "Intact",    "Second_Half"): (1.83, 1.45, 2.22),
    ("unconscious_aware",   "Scrambled", "First_Half"):  (1.29, 0.91, 1.67),
    ("unconscious_aware",   "Scrambled", "Second_Half"): (1.25, 0.87, 1.63),
    ("unconscious_unaware", "Intact",    "First_Half"):  (1.83, 1.46, 2.21),
    ("unconscious_unaware", "Intact",    "Second_Half"): (2.10, 1.72, 2.48),
    ("unconscious_unaware", "Scrambled", "First_Half"):  (2.19, 1.82, 2.55),
    ("unconscious_unaware", "Scrambled", "Second_Half"): (2.13, 1.76, 2.51),
}

fig, ax = plt.subplots(figsize=(9, 6))

for center, awareness in enumerate(AWARENESS_ORDER):
    for ref, color in COLORS.items():
        base = center + REF_DODGE[ref]
        xs = [base - HALF_OFFSET, base + HALF_OFFSET]
        half_colors = [color, darken(color)]

        # model EMMs + 95% CI (asymmetric error from the CI bounds)
        means = [EMMS[(awareness, ref, h)][0] for h in HALVES]
        lo = [EMMS[(awareness, ref, h)][1] for h in HALVES]
        hi = [EMMS[(awareness, ref, h)][2] for h in HALVES]
        yerr = [np.subtract(means, lo), np.subtract(hi, means)]
        ax.errorbar(xs, means, yerr=yerr, color=color, capsize=4, zorder=1)
        ax.scatter(xs, means, color=half_colors, zorder=2)

ax.set_xticks(range(len(AWARENESS_ORDER)))
ax.set_xticklabels([a.replace("_", " ") for a in AWARENESS_ORDER])
ax.set_xlabel("Awareness")
ax.set_ylabel("Cross-phase NSS (model EMM, 95% CI)")
ax.axhline(0, color="grey", lw=0.8, ls="--")

# Legend: one entry per reference map + a light/dark = half guide.
handles = [plt.Line2D([], [], color=c, marker="o", label=r) for r, c in COLORS.items()]
handles += [
    plt.Line2D([], [], color="grey", marker="o", ls="", label="First half (lighter)"),
    plt.Line2D([], [], color=darken("grey"), marker="o", ls="", label="Second half (darker)"),
]
ax.legend(handles=handles)
fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
