"""The cross-phase effect plotted as the difference it is actually tested on.

The jamovi EMM plot draws one point per cell with its own error bar, which invites the
reading "the bars do not overlap, so the effect is significant". That reading fails here:
the bars carry between-participant and between-image variance, but Intact and Scrambled
are measured on the same participants and the same images, so most of that cancels in the
difference. In the replication set the two cell estimates correlate at 0.47 (aware) and
0.77 (unaware), which is why the unaware effect is the significant one despite bars that
overlap more. This plots the contrast itself, with the model's own confidence interval,
so the picture and the test answer the same question.

Run CrossPhaseModel.py first -- the estimates are read from what it saved, never refitted.

Reads:  Analysis_Results/<dataset>/NSS_whole[_suffix]/CrossPhaseModel_<label>.csv  (CrossPhaseModel.py)
Writes: Figures/<dataset>/nss_analyses/NSS_CrossPhase_Difference_<label>*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

# === CONFIG ===
# Which dataset to plot. It has to be set before Settings is imported (Settings reads
# EYESPY_DATASET on import and would otherwise ask), so it sits above the imports below.
import os
DATASET = "data_rep"   # "data_pilot" | "data_rep"
os.environ.setdefault("EYESPY_DATASET", "rep" if DATASET == "data_rep" else "pilot")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
import pandas as pd
from Scripts.Analysis.NSS import NSSPaths
from Settings import _SUFFIX as DATASET_SUFFIX, FIGURES_ROOT  # "_pilot" or "_rep", per EYESPY_DATASET

# Must match RESULTS_LABEL in CrossPhaseModel.py, so the plot cannot silently pick up a
# different run than the one you just fitted.
RESULTS_LABEL = "primary"

CONFIDENCE = 0.95

TRIAL_SET = NSSPaths.ask_trial_set()
BLINK_MODE = NSSPaths.ask_blink_mode()

# No Early/Late dimension here, so always the whole-mode results.
RESULTS_DIR = NSSPaths.paths_for("whole", TRIAL_SET, BLINK_MODE)["OUTPUT_DIR"]
INPUT_FILE = RESULTS_DIR / f"CrossPhaseModel_{RESULTS_LABEL}.csv"

OUTPUT_DIR = Path(f"{FIGURES_ROOT}/nss_analyses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_SUFFIX = NSSPaths.TRIAL_SET_SUFFIX[TRIAL_SET] + NSSPaths.BLINK_SUFFIX[BLINK_MODE] + DATASET_SUFFIX
OUTPUT_PLOT = OUTPUT_DIR / f"NSS_CrossPhase_Difference_{RESULTS_LABEL}{_SUFFIX}.png"

PALETTE = ['#edf8fb', '#b3cde3', '#648fff', '#785ef0']
POINT_COLOR = PALETTE[3]

GROUP_LABELS = {
    "conscious_aware": "Conscious Aware\n(PAS 2-3)",
    "unconscious_aware": "Unconscious Aware\n(PAS 2-3)",
    "unconscious_unaware": "Unconscious Unaware\n(PAS 0)",
}

# --- Load the saved model results
if not INPUT_FILE.exists():
    raise SystemExit(
        f"\n{INPUT_FILE} does not exist.\n"
        f"Run Scripts/Analysis/Statistics/CrossPhaseModel.py first "
        f"(with RESULTS_LABEL = {RESULTS_LABEL!r}).\n"
    )

results = pd.read_csv(INPUT_FILE)
effects = results[results["table"] == "simple_effect"].reset_index(drop=True)
print(f"Reading  {INPUT_FILE}")

# --- Confidence interval from the model's own estimate, SE and Satterthwaite df, so the
# interval and the p-value are the same test drawn two ways.
critical_t = stats.t.ppf(1 - (1 - CONFIDENCE) / 2, effects["df_residual"])
effects["ci_low"] = effects["estimate"] - critical_t * effects["standard_error"]
effects["ci_high"] = effects["estimate"] + critical_t * effects["standard_error"]

# --- Plot
figure, axes = plt.subplots(figsize=(7, 5.5))
x_positions = range(len(effects))

axes.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)

for x, effect in zip(x_positions, effects.itertuples()):
    axes.errorbar(
        x, effect.estimate,
        yerr=[[effect.estimate - effect.ci_low], [effect.ci_high - effect.estimate]],
        fmt="o", markersize=11, capsize=6, linewidth=2, zorder=3,
        color=POINT_COLOR,
        # Filled if it survived the Tree-BH correction, hollow if it did not.
        markerfacecolor=POINT_COLOR if effect.survives else "white",
        markeredgewidth=2,
    )
    verdict = "survives Tree-BH" if effect.survives else "does not survive"
    axes.annotate(
        f"{effect.estimate:+.2f}\np = {effect.p_value:.4f}\n{verdict}",
        xy=(x, effect.ci_high), xytext=(0, 12), textcoords="offset points",
        ha="center", va="bottom", fontsize=9,
    )

axes.set_xticks(list(x_positions))
axes.set_xticklabels([GROUP_LABELS.get(term, term) for term in effects["term"]])
axes.set_xlim(-0.6, len(effects) - 0.4)
axes.set_ylabel("NSS difference  (Intact - Scrambled)")
axes.set_title(f"Cross-phase effect per awareness state\n"
               f"model estimate with {CONFIDENCE:.0%} CI, Satterthwaite df",
               fontsize=11)
axes.spines[["top", "right"]].set_visible(False)

# Headroom for the annotations, which sit above the upper CI.
axes.set_ylim(min(effects["ci_low"].min(), 0) - 0.12, effects["ci_high"].max() + 0.26)

figure.tight_layout()
figure.savefig(OUTPUT_PLOT, dpi=300)
print(f"Saved    {OUTPUT_PLOT}")

for effect in effects.itertuples():
    print(f"  {effect.term:<22} {effect.estimate:+.3f}  "
          f"[{effect.ci_low:+.3f}, {effect.ci_high:+.3f}]  p = {effect.p_value:.4f}")
