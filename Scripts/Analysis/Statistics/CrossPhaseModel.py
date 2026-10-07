"""Fits the registered cross-phase mixed model and applies the Tree-BH correction.

Replaces the jamovi run plus the by-hand correction arithmetic with one command, so a
refit (different groups, slope reading, trial set or blink mode) cannot silently leave
the reported p-values behind. _lmer.R does the fitting, because jamovi's GAMLj is a
wrapper over lme4 + lmerTest and only those give Satterthwaite df. This file owns every
analysis decision and the correction.

Every choice is a named constant in the DECISIONS block below. Nothing is prompted.

Reads:  Analysis_Results/<dataset>/NSS_<mode>[_suffix]/NSS_CrossPhase_LongFormat.csv   (NSS.py)
Writes: Analysis_Results/<dataset>/NSS_<mode>[_suffix]/CrossPhaseModel_<label>.csv
        Analysis_Results/<dataset>/NSS_<mode>[_suffix]/CrossPhaseModel_<label>.txt
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Work from the project root whatever launched this -- terminal, VS Code's play button,
# an interactive window -- because NSSPaths returns paths relative to it.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Dataset toggle. Set before Settings is imported, because Settings reads
# EYESPY_DATASET at import time and would otherwise ask. setdefault, not assignment, so a
# one-off `EYESPY_DATASET=rep python3 ...` still wins.
#   "data_pilot" the pilot set in Data.nosync/pilot/,   Settings.py + Settings_pilot.py on top
#   "data_rep"   the replication set in Data.nosync/rep/, Settings.py + Settings_rep.py on top
DATASET = "data_pilot"   # "data_pilot" | "data_rep"
os.environ.setdefault("EYESPY_DATASET", "rep" if DATASET == "data_rep" else "pilot")

from Scripts.Analysis.NSS import NSSPaths

# ============================================================================
# DECISIONS
# ============================================================================

# Which dataset: the DATASET toggle sits above the imports, for the reason given there.

# Awareness levels to keep. All three is the full registered model; the two unconscious
# levels is the unconscious-only variant.
AWARENESS_GROUPS = ["unconscious_aware", "unconscious_unaware"]

# The registered model. Random slopes sit on ReferenceMap (= disambiguator type) on both
# grouping factors; delete "+ (1 + ReferenceMap | Image)" for the slope-reading
# sensitivity analysis that puts the slope on Participant only.
MODEL_FORMULA = (
    "NSS ~ Awareness * ReferenceMap"
    " + (1 + ReferenceMap | Participant)"
    " + (1 + ReferenceMap | Image)"
)

# How factors enter the design matrix. This changes nothing about which hypotheses are
# tested, but it does change the path the optimizer takes, and on this data that matters:
# under "treatment" coding bobyqa stops at a singular boundary with the Image intercept
# variance collapsed to zero (logLik -8911.1), while under "sum" coding it converges
# cleanly (-8884.6). "sum" is what jamovi's GAMLj uses, so it reproduces the jamovi run;
# "treatment" is plain R's default.
FACTOR_CODING = "sum"

# The jamovi run used bobyqa, which converges here given FACTOR_CODING = "sum". The
# singularity flag and log-likelihood are printed either way, so a change stays visible.
OPTIMIZER = "bobyqa"

# Tree-BH (Bogomolov et al. 2021), registered as Figure 6. Level 1 is a BH family of two,
# tested at FALSE_DISCOVERY_RATE. Level 2 -- the simple effect of disambiguator type
# within each awareness group -- hangs off the interaction node ONLY, so it is tested only
# if that node survives, and then at a stricter threshold (see below).
FALSE_DISCOVERY_RATE = 0.05
LEVEL1_TESTS = ["Awareness:ReferenceMap", "ReferenceMap"]
LEVEL2_PARENT = "Awareness:ReferenceMap"

# Names the saved files, so a sensitivity run does not overwrite the primary one.
RESULTS_LABEL = "primary"

# Optional self-check: raw p-values from a previous run, to confirm nothing has drifted.
# Set to None to switch it off. These are the jamovi values for the unconscious-only model
# on the replication set, whole Mooney window, interpolated blinks, all trials.
PREVIOUS_RUN_PVALUES = {
    "Awareness": 0.0319,
    "ReferenceMap": 0.4609,
    "Awareness:ReferenceMap": 0.0009,
    "unconscious_aware": 0.0318,
    "unconscious_unaware": 0.0092,
}


# ============================================================================
# Printing, captured so the same text can be saved
# ============================================================================

report_lines = []


def say(line=""):
    """Print a line and keep it for the saved report."""
    print(line)
    report_lines.append(line)


# ============================================================================
# The correction
# ============================================================================

def benjamini_hochberg(p_values):
    """BH-adjusted p-values, in the same order as the input.

    BH is a step-up procedure, so the members of a family are coupled. Adjusting the
    p-values folds the ranking in, which lets every member be compared to the same single
    threshold: a test survives when its adjusted p-value is at or below it.
    """
    n_tests = len(p_values)
    smallest_first = sorted(range(n_tests), key=lambda i: p_values[i])
    adjusted = [0.0] * n_tests
    running_minimum = 1.0
    for rank, i in reversed(list(enumerate(smallest_first, start=1))):
        running_minimum = min(running_minimum, p_values[i] * n_tests / rank)
        adjusted[i] = running_minimum
    return adjusted


def report_family(test_names, p_values, threshold):
    """Print one BH family as a table, and return a row per test for the saved CSV.

    "its bar" is the raw p-value this test needed at the rank it actually holds,
    i.e. threshold x rank / number of tests.
    """
    adjusted = benjamini_hochberg(p_values)
    n_tests = len(p_values)
    rank_of_test = {i: rank for rank, i in enumerate(
        sorted(range(n_tests), key=lambda i: p_values[i]), start=1)}

    say(f"  {'':<24} {'raw p':>8} {'BH-adj':>8} {'threshold':>10} {'its bar':>9}")
    rows = []
    for i, name in enumerate(test_names):
        survives = adjusted[i] <= threshold
        its_bar = threshold * rank_of_test[i] / n_tests
        say(f"  {name:<24} {p_values[i]:>8.4f} {adjusted[i]:>8.4f} "
            f"{threshold:>10.4f} {its_bar:>9.4f}   "
            f"{'SURVIVES' if survives else 'fails'}")
        rows.append({"term": name, "p_adjusted": adjusted[i], "threshold": threshold,
                     "its_bar": its_bar, "survives": survives})
    return rows


# ============================================================================
# The fit
# ============================================================================

def fit_model_in_r(data):
    """Fit MODEL_FORMULA on `data` via _lmer.R and return its tidy results table."""
    r_script = Path(__file__).with_name("_lmer.R")
    with tempfile.TemporaryDirectory() as working_directory:
        model_input = Path(working_directory) / "model_input.csv"
        model_results = Path(working_directory) / "model_results.csv"
        data.to_csv(model_input, index=False)
        subprocess.run(
            ["Rscript", str(r_script), str(model_input), str(model_results),
             MODEL_FORMULA, OPTIMIZER, FACTOR_CODING],
            check=True,  # R's own convergence warnings pass through to the terminal
        )
        return pd.read_csv(model_results)


nss_paths = NSSPaths.select()
all_rows = pd.read_csv(nss_paths["CROSS_CSV"])

# Fail loudly rather than quietly fitting a smaller model than you asked for.
groups_in_file = sorted(all_rows["Awareness"].unique())
groups_missing = [g for g in AWARENESS_GROUPS if g not in groups_in_file]
if groups_missing:
    raise SystemExit(
        f"\nAWARENESS_GROUPS asks for {groups_missing}, which this file does not have.\n"
        f"It contains only {groups_in_file}.\n"
    )

model_data = all_rows[all_rows["Awareness"].isin(AWARENESS_GROUPS)]

say(f"\nReading  {nss_paths['CROSS_CSV']}")
say(f"Data     {len(model_data)} rows, {model_data['Participant'].nunique()} "
    f"participants, {model_data['Image'].nunique()} images")
for group, n_rows in model_data["Awareness"].value_counts().items():
    say(f"         {group:<22}{n_rows:>6} rows")
say(f"Model    {MODEL_FORMULA}")
say(f"Engine   lme4 + lmerTest, REML, {OPTIMIZER}, {FACTOR_CODING} contrasts, "
    f"Satterthwaite df")

results = fit_model_in_r(model_data)
omnibus_tests = results[results["table"] == "omnibus"].set_index("term")
simple_effects = results[results["table"] == "simple_effect"].set_index("term")
fit_diagnostics = results[results["table"] == "fit"].set_index("term")["estimate"]

say(f"Fit      log-likelihood {fit_diagnostics['log_likelihood']:.2f}, "
    f"{'SINGULAR' if fit_diagnostics['is_singular'] else 'not singular'}\n")

if fit_diagnostics["is_singular"]:
    say("!! Singular fit: a variance component sat on the boundary, so the optimizer")
    say("   stopped somewhere the df cannot be trusted. Compare the log-likelihood")
    say("   against another OPTIMIZER or FACTOR_CODING before reporting anything.\n")


# ============================================================================
# The report
# ============================================================================

say("Fixed effects omnibus tests")
say(f"  {'':<24} {'F':>8} {'df':>4} {'df(res)':>9} {'p':>8}")
for name, test in omnibus_tests.iterrows():
    say(f"  {name:<24} {test['statistic']:>8.2f} {test['df']:>4.0f} "
        f"{test['df_residual']:>9.2f} {test['p_value']:>8.4f}")

say("\nSimple effect of disambiguator type (Intact - Scrambled) within each group")
say(f"  {'':<24} {'estimate':>8} {'SE':>7} {'t':>7} {'df':>8} {'p':>8}")
for name, effect in simple_effects.iterrows():
    say(f"  {name:<24} {effect['estimate']:>8.3f} {effect['standard_error']:>7.3f} "
        f"{effect['statistic']:>7.2f} {effect['df_residual']:>8.2f} "
        f"{effect['p_value']:>8.4f}")

say(f"\nTree-BH level 1   (family of {len(LEVEL1_TESTS)}, q = {FALSE_DISCOVERY_RATE})")
level1_p_values = [omnibus_tests.loc[name, "p_value"] for name in LEVEL1_TESTS]
level1_rows = report_family(LEVEL1_TESTS, level1_p_values, FALSE_DISCOVERY_RATE)
tree_rows = list(level1_rows)

# Level 2 exists only under the interaction node, so a failed parent ends the analysis.
parent_survived = next(r["survives"] for r in level1_rows if r["term"] == LEVEL2_PARENT)
if parent_survived:
    # Selection-adjusted threshold: the price of having chosen this branch of the tree.
    n_survived_level1 = sum(r["survives"] for r in level1_rows)
    level2_threshold = FALSE_DISCOVERY_RATE * n_survived_level1 / len(LEVEL1_TESTS)

    say(f"\nTree-BH level 2   (family of {len(simple_effects)}, "
        f"q2 = {FALSE_DISCOVERY_RATE} x {n_survived_level1}/{len(LEVEL1_TESTS)} "
        f"= {level2_threshold})")
    level2_names = list(simple_effects.index)
    level2_p_values = [simple_effects.loc[name, "p_value"] for name in level2_names]
    tree_rows += report_family(level2_names, level2_p_values, level2_threshold)
else:
    say(f"\n  '{LEVEL2_PARENT}' did not survive level 1, so the simple effects are never")
    say("  tested and neither can survive. Nothing further to report.")

if PREVIOUS_RUN_PVALUES:
    say("\nCheck against the p-values recorded from the previous run")
    p_value_by_name = {**omnibus_tests["p_value"], **simple_effects["p_value"]}
    for name, recorded in PREVIOUS_RUN_PVALUES.items():
        if name not in p_value_by_name:
            continue
        now = p_value_by_name[name]
        matches = abs(now - recorded) < 0.0002  # both were rounded to 4 decimal places
        say(f"  {name:<24} recorded {recorded:>8.4f}   now {now:>8.4f}   "
            f"{'ok' if matches else 'DIFFERS'}")


# ============================================================================
# Save, into the same versioned folder the input came from
# ============================================================================

results_csv = nss_paths["OUTPUT_DIR"] / f"CrossPhaseModel_{RESULTS_LABEL}.csv"
report_txt = nss_paths["OUTPUT_DIR"] / f"CrossPhaseModel_{RESULTS_LABEL}.txt"

saved = results.merge(pd.DataFrame(tree_rows), on="term", how="left")
saved.to_csv(results_csv, index=False)
report_txt.write_text("\n".join(report_lines) + "\n")

print(f"\nSaved  {results_csv}")
print(f"Saved  {report_txt}")
