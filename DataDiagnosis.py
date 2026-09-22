"""Per-participant inclusion audit: how much data each one brings, and how much is discarded.

Data leaves this analysis at six different points — the manual exclusions, calibration,
the behavioural trial filters, Stage-1 gaze filtering, the per-image subject threshold and
the per-cell image threshold — spread over Settings.py, three scripts and two stages. A
participant missing from the results says nothing about which of those removed them.

The table answers two questions per participant, and nothing else. How much data does this
person bring (Eligible: the post-intact Mooney trials surviving the behavioural filters)
and how much of it survives to the model (Analysed, and Discarded as the difference)? And
when a cell contributes nothing, what removed it (Verdict: the FIRST of the six points that
did, colour-coded so the dominant cause across the sample is visible at a glance)?

A participant is one block of the table, separated by a rule, holding a row per awareness
cell — that cell is the unit the NSS threshold acts on, so it has to be the row. Values
that belong to the session rather than the cell (the fitted alpha, the PAS distribution and
the three recording-quality columns) are written once per block, across its rows.

Every session lists BOTH of its awareness cells, including the ones that hold no trials at
all and the ones of a manually excluded session, so a missing cell is always visible as a
row with a verdict rather than as an absence. That is also what makes the PAS column worth
reading: it is the session's full response distribution, PAS 1 included, so a cell that is
thin because the participant rarely reported that awareness level says so on its own row.

The raw files (.asc + .mat) carry Eligible and every upstream verdict, so the script runs
standalone. Analysed comes from the pipeline's outputs; when those are absent it still
runs, leaves Analysed blank, marks the cells ELIGIBLE instead of IN, and says so — an
eligibility ceiling rather than a membership list, since raw data cannot see gaze loss.
The intermediate steps that are no longer columns (trials surviving Stage 1, trials
scored) are still computed, as the consistency check that catches mismatched outputs.
Outputs older than their own inputs are reported as stale rather than trusted.

Strictly read-only: it recomputes nothing, creates no folders, and writes only the figure.

DATASET below is CompleteRun.py's toggle and is the only switch a normal run needs: it
points the recordings, the parquet, analysesresults[_rep]/, the figure and the EXCLUDE_*
lists at one dataset together. Auditing data outside the repository takes --data, and then
APPLY_EXCLUSIONS off (or --no-exclusions), because Settings.py's EXCLUDE_* lists name the
participants of a dataset whose numbering another dataset repeats:

    python3 DataDiagnosis.py --data ~/Desktop/PilotTwo --no-exclusions --out ~/Desktop/audit.png
    python3 DataDiagnosis.py --help          # every path it can be told about

Reads:  <data>/raw/s_<SESSION>_<PID>.asc                          (blinks, calibration, recording spans)
        <data>/behavioural/expdata_<SESSION>_<PID>.mat            (trials, PAS, blocks,
                                                                   threshold-fitting alpha)
        Settings.py                                               (exclusions and thresholds)
        default <data> is data[_rep]/my_dataset/, following DATASET
        optional, for Analysed and Discarded:
        DataQualityChecks[_rep]/blink_spatial_filtering.csv        (CompleteRun.py)
        data[_rep]/NSS_all_fixations_clean.parquet                 (NSSExporter.py)
        analysesresults[_rep]/NSS_<mode>/NSS_crossphase_descriptives.pkl (NSS.py)
        analysesresults[_rep]/NSS_<mode>/NSS_CrossPhase_LongFormat.csv   (NSS.py)
Writes: Figures[_rep]/DataDiagnosis.png                             (or --out)
"""

#%% Imports and configuration
from pathlib import Path
from collections import Counter
import argparse
import mmap
import os
import pickle
import re
import sys

#%% Run switches: what a plain "Run" on this file uses (the command line overrides both)
# DATASET is CompleteRun.py's toggle, and it has to be set before Settings.py is imported
# because that is what Settings.py branches on. Setting it here is what keeps every path
# consistent in one move: the recordings, the parquet, analysesresults[_rep]/, the figure
# and - the reason APPLY_EXCLUSIONS can default to True - the matching EXCLUDE_* lists.
DATASET          = "data_rep"   # "data" | "data_rep", exactly as in CompleteRun.py
APPLY_EXCLUSIONS = True         # False -> judge every cell on its own data instead
os.environ["EYESPY_DATASET"] = "rep" if DATASET == "data_rep" else ""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # project root, for the imports below
from Settings import (RAW_DATA_DIR, BEHAVIOURAL_DIR, EVENTS_CLEANED_DIR, FIX_FILE,
                      ANALYSES_ROOT, data_quality_folder, SECTION_TO_BLOCK, TRIAL_SET,
                      MOONEY_SPLIT,
                      INTERPOLATE_BLINKS, EXCLUDE_SUBJECTS, EXCLUDE_SESSIONS, EXCLUDE_BLOCKS,
                      N_EXPERIMENT_BLOCKS,
                      VALIDATION_ACCURACY_AVG_THRESHOLD, VALIDATION_ACCURACY_MAX_THRESHOLD,
                      CENTER_RADIUS_DG,
                      MIN_VIEWINGS_PER_IMAGE_CROSS, MIN_IMAGES_PER_CELL_CROSS,
                      FILTER_PALETTE, _SUFFIX as DATASET_SUFFIX)
from Scripts.Analysis.NSS.NSSPaths import TRIAL_SET_SUFFIX, BLINK_SUFFIX

# The results folder for the modes in Settings.py. Built from ANALYSES_ROOT and NSSPaths'
# own suffix maps rather than NSSPaths.select(), which would create the folder as a side
# effect. ANALYSES_ROOT follows DATASET, so a data_rep run reads analysesresults_rep/.
BLINK_MODE    = "interp" if INTERPOLATE_BLINKS else "filter"
DEFAULT_CROSS = Path(ANALYSES_ROOT, f"NSS_{MOONEY_SPLIT}"
                                    f"{TRIAL_SET_SUFFIX[TRIAL_SET]}{BLINK_SUFFIX[BLINK_MODE]}")

#%% Where the data is: the switches above, then the command line, then Settings.py
def parse_args():
    """Resolve every path this script reads and writes.

    Nothing has to be given: DATASET above already points every path at one dataset of
    the repository. The options are for data that lives elsewhere - point --data at a
    folder anywhere and its raw/ and behavioural/ subfolders are picked up (or the folder
    itself, if it holds the recordings directly), or name any single path outright.
    """
    parser = argparse.ArgumentParser(
        description="Audit which participants the NSS analysis keeps, and what removes the rest.",
        epilog="examples:\n"
               "  python3 DataDiagnosis.py\n"
               "  python3 DataDiagnosis.py --data ~/Desktop/PilotTwo --no-exclusions\n"
               "  python3 DataDiagnosis.py --data /Volumes/Backup/study --out ~/Desktop/audit.png",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--data", metavar="DIR",
        help="dataset folder holding raw/ and behavioural/ "
             f"(default: {Path(RAW_DATA_DIR).parent}, from DATASET above)")
    add("--raw", metavar="DIR", help="the .asc recordings, if they are not in <data>/raw")
    add("--behavioural", metavar="DIR", help="the .mat files, if they are not in <data>/behavioural")
    add("--parquet", metavar="FILE", help=f"fixations parquet, for the exact verdict (default: {FIX_FILE})")
    add("--results", metavar="DIR", help=f"NSS results folder, for the exact verdict (default: {DEFAULT_CROSS})")
    # Follows the dataset switch, like the plotting scripts: a data_rep run writes to
    # Figures_rep/ so it cannot overwrite the pilot's figure (and vice versa).
    add("--out", metavar="FILE", default=f"Figures{DATASET_SUFFIX}/DataDiagnosis.png",
        help="figure to write (default: %(default)s)")
    add("--no-exclusions", action="store_true", default=not APPLY_EXCLUSIONS,
        help="ignore Settings.py's EXCLUDE_* lists, which name one dataset's participants "
             f"(default: {not APPLY_EXCLUSIONS}, from APPLY_EXCLUSIONS above)")
    # An interactive window passes the kernel its own arguments, so only a real command
    # line is parsed; run this as a cell and the defaults above apply.
    return parser.parse_args([] if "ipykernel" in sys.modules else None)

ARGS = parse_args()
DATA_DIR = Path(ARGS.data).expanduser() if ARGS.data else None

def resolve(explicit, name, default):
    """A folder given outright, else found inside --data, else the Settings.py default."""
    if explicit:
        return Path(explicit).expanduser()
    if DATA_DIR:
        return DATA_DIR / name if (DATA_DIR / name).is_dir() else DATA_DIR
    return Path(default)

RAW_DIR   = resolve(ARGS.raw, "raw", RAW_DATA_DIR)
BEH_DIR   = resolve(ARGS.behavioural, "behavioural", BEHAVIOURAL_DIR)
# Pipeline outputs belong to one dataset. Once the recordings are pointed somewhere else,
# reusing this repo's outputs would score a different sample's participants against them,
# so they are looked for beside the given dataset (mirroring data/my_dataset -> data/) and
# otherwise not used at all; naming one outright always wins.
DATASET_ROOT = DATA_DIR or (RAW_DIR.parent if ARGS.raw else None)
FIX_PATH  = (Path(ARGS.parquet).expanduser() if ARGS.parquet else
             DATASET_ROOT.parent / Path(FIX_FILE).name if DATASET_ROOT else Path(FIX_FILE))
CROSS_DIR = (Path(ARGS.results).expanduser() if ARGS.results else
             DATASET_ROOT.parent / DEFAULT_CROSS if DATASET_ROOT else DEFAULT_CROSS)
OUT_PNG   = Path(ARGS.out).expanduser()
CROSS_PKL, CROSS_CSV = CROSS_DIR / "NSS_crossphase_descriptives.pkl", CROSS_DIR / "NSS_CrossPhase_LongFormat.csv"

if not RAW_DIR.is_dir():
    raise SystemExit(f"no folder at {RAW_DIR.resolve()}\n"
                     "Point the script at the data with --data DIR (or --raw DIR); --help lists the options.")

# A viewing yields one row per reference map, and in "halves" mode one per Mooney half, so
# this many rows per viewing count towards MIN_IMAGES_PER_CELL_CROSS.
MIN_VIEWINGS = int(np.ceil(MIN_IMAGES_PER_CELL_CROSS / (2 if MOONEY_SPLIT == "halves" else 1)))

# One colour per outcome, in cascade order. The key is also the text in the Verdict column,
# so the legend, the cell colours and the terminal summary can never drift apart.
INCLUDED   = "included"
ELIGIBLE   = "eligible (pipeline not run)"
THIN_TRIAL = f"under {MIN_VIEWINGS} eligible trials"
NO_FIX     = "all fixations lost in Stage 1"
THIN_FIX   = f"under {MIN_VIEWINGS} trials survived Stage 1"
NO_BLOCKS  = "no trials left after block exclusions"
NO_PAS     = "no trials with this PAS response"
ALL_UNUSED = "every trial of this PAS unanswered or repeated"
NO_IMAGE   = f"no image reached {MIN_VIEWINGS_PER_IMAGE_CROSS} viewings"
THIN_CELL  = f"scored, but under {MIN_IMAGES_PER_CELL_CROSS} valid scores"
REASONS = {
    INCLUDED:                     "#a8dda8",  # green
    ELIGIBLE:                     "#d6ecd6",  # pale green
    "manual: subject excluded":   "#f28b82",  # red
    "manual: session excluded":   "#f6bd60",  # orange
    "every calibration failed":   "#ffe08a",  # yellow
    "no behavioural file":        "#c9b8a8",  # taupe
    NO_BLOCKS:                    "#b0aae0",  # periwinkle
    NO_PAS:                       "#c0c0c0",  # grey
    ALL_UNUSED:                   "#e8c8a0",  # tan
    THIN_TRIAL:                   "#e0a3e8",  # purple
    NO_FIX:                       "#79c7e3",  # blue
    THIN_FIX:                     "#a8e0d8",  # teal
    NO_IMAGE:                     "#d3b8f0",  # lilac
    THIN_CELL:                    "#f0b8d0",  # pink
}

# The PAS bar: the scale in order, then "na" for a trial whose rating is missing, which is
# part of the session's 100% like any other answer. Colours come from Settings.FILTER_PALETTE
# (light = low) so a participant's bar here and in ResponseDistributions.py read the same.
PAS_LEVELS  = (0, 1, 2, 3, "na")
PAS_COLOURS = (*FILTER_PALETTE, "#c0c0c0")
PAS_LABELS  = ("PAS 0  no experience", "PAS 1  brief glimpse", "PAS 2  almost clear",
               "PAS 3  clear", "no answer")

# Settings.py keys participants as ints, the filenames as strings: normalise once, as
# apply_behavioral_filters_and_save does, or the exclusions silently never match.
# These lists name the participants of the dataset the repository is set up for, and
# another dataset numbers its participants from the same range, so APPLY_EXCLUSIONS (or
# --no-exclusions) drops them: every cell is then judged on its own data, which is what a
# fresh sample needs.
EXCL_SUBJ  = set() if ARGS.no_exclusions else {str(p) for p in EXCLUDE_SUBJECTS}
EXCL_SESS  = {} if ARGS.no_exclusions else {str(k): v for k, v in EXCLUDE_SESSIONS.items()}
EXCL_BLOCK = {} if ARGS.no_exclusions else {str(k): v for k, v in EXCLUDE_BLOCKS.items()}

# Which .mat sections count, mirroring TRIAL_SET (Practice never enters the analysis).
WANTED_BLOCKS = {"all": {"Experiment", "Extra"},
                 "experiment": {"Experiment"}, "extra": {"Extra"}}[TRIAL_SET]

#%% Raw readers: the funnel's first column, and the quality context
def read_trials(mat_path):
    """Return (trials, alpha) for one behavioural .mat, or (None, None) if unreadable.

    Mirrors load_behavioural_from_mat (section walk, ghost-trial skip) but reads the fields
    directly: the audit needs IsIntactDisambiguation, which is not in Settings.MAT_FIELD_MAP,
    and no other consumer of that map should have to grow a column for this script. The
    alpha comes from the same load because a .mat carries a per-frame Alpha array on every
    trial, so opening these files twice for one number is the expensive way to get it.
    """
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        expdata = next(v for k, v in mat.items() if k.lower() == "expdata")
    except Exception:
        return None, None

    def field(trial, name):
        """One scalar field, with MATLAB's empty placeholders normalised to None."""
        value = getattr(trial, name, None)
        return None if isinstance(value, np.ndarray) and value.size == 0 else value

    trials = []
    for section, block in SECTION_TO_BLOCK.items():
        if block not in WANTED_BLOCKS or not hasattr(expdata, section):
            continue
        struct = getattr(expdata, section)
        section_trials = []
        for trial in (struct if isinstance(struct, np.ndarray) else [struct]):
            if field(trial, "TrialNum") is None:      # ghost trial: an empty placeholder row
                continue
            row = {k: field(trial, src) for k, src in
                   [("block", "BlockNum"), ("pas", "response_PAS_Q"),
                    ("answered", "did_answer_PAS_Q"),
                    ("answered_pls", "did_answer_PLS_Q"),
                    ("fixfail", "NumRepetitionFixationFail"),
                    ("intact", "IsIntactDisambiguation")]}
            row["block_type"] = block
            row["pre_break"] = False
            section_trials.append(row)

        # Preregistration exclusion (b) drops the trial BEFORE a broken fixation, not the
        # flagged one. Within a section the rows are in trial order, so the trial before is
        # simply the previous row - the same (block_type, trial_number - 1) the pipeline
        # uses. A failure on a section's first trial has no predecessor and drops nothing.
        for i, row in enumerate(section_trials):
            if (row["fixfail"] or 0) > 0 and i > 0:
                section_trials[i - 1]["pre_break"] = True
        trials.extend(section_trials)

    # Without the intact flag every trial would look scrambled and every cell would be
    # reported as too thin, which reads like a result rather than a missing field.
    if trials and all(t["intact"] is None for t in trials):
        raise SystemExit(f"{Path(mat_path).name}: no IsIntactDisambiguation field in the "
                         "expdata trials, so post-intact viewings cannot be counted.")
    return trials, read_fitted_alpha(expdata)


def read_fitted_alpha(expdata):
    """The image alpha the threshold fitting settled on, or None if it never did.

    The fitting runs blocks of increasing alpha until one reaches criterion, and that
    block is the one flagged `passed`; its `alpha_image` is what every Experiment and
    Extra trial is then shown at (checked against the trials' own per-frame Alpha ramp on
    this dataset: they agree in all 29 sessions). Blocks that never ran are stored as
    empty placeholders, so they are skipped rather than read as alpha 0.

    Returns (alpha, passed). A session whose fitting ended without a passing block still
    reports the last alpha it reached, flagged as not passed, since a missing number would
    hide the more interesting fact that the staircase never converged.
    """
    blocks = getattr(expdata, "Blocks_ThresholdFitting", None)
    if blocks is None:
        return None, False
    attempted = []
    for block in (blocks if isinstance(blocks, np.ndarray) else [blocks]):
        alpha = getattr(block, "alpha_image", None)
        if alpha is None or (isinstance(alpha, np.ndarray) and alpha.size == 0):
            continue                      # a fitting block the session never needed to run
        attempted.append((float(alpha), bool(getattr(block, "passed", 0))))
    passed = [alpha for alpha, ok in attempted if ok]
    if passed:
        return passed[-1], True
    return (attempted[-1][0], False) if attempted else (None, False)

BLINK      = re.compile(rb"EBLINK\s+\S+\s+(\d+)\s+(\d+)")
BLOCK_MARK = re.compile(rb"\n(START|END)\s+(\d+)")
VAL_TIME   = re.compile(rb"MSG\s+(\d+)\s+!CAL VALIDATION")
VAL_ERROR  = re.compile(rb"VALIDATION\s+\S+\s+\S+\s+\S+\s+\w+\s+ERROR\s+([\d.]+)\s+avg\.\s+([\d.]+)\s+max")

def scan_asc(asc_path):
    """Blink burden and calibration loss for one recording, in a single mmap pass.

    Returns (blink % of recorded time, n_failed, n_validations, % of recorded time Stage 1
    discards for bad calibration). The calibration figure reproduces
    filter_and_report_validations: a validation missing either threshold invalidates
    everything up to the next validation.
    """
    with open(asc_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        blinks  = [(int(a), int(b)) for a, b in BLINK.findall(raw)]
        marks   = [(kind.decode(), int(t)) for kind, t in BLOCK_MARK.findall(raw)]
        v_times = [int(t) for t in VAL_TIME.findall(raw)]
        v_error = [(float(avg), float(mx)) for avg, mx in VAL_ERROR.findall(raw)]

    # Recorded stretches: every START paired with the END that closes it
    blocks = [(marks[i][1], marks[i + 1][1]) for i in range(len(marks) - 1)
              if marks[i][0] == "START" and marks[i + 1][0] == "END"]
    recorded = sum(end - start for start, end in blocks)
    if not recorded:
        return float("nan"), 0, 0, float("nan")

    def overlap(spans):
        """Total time these spans share with the recorded stretches."""
        return sum(max(0, min(e, se) - max(s, ss)) for ss, se in blocks for s, e in spans)

    # A validation and its accuracy come from the same line, so the lists align by index;
    # guard anyway, since a truncated recording could cut one of them short.
    n_val  = min(len(v_times), len(v_error))
    bounds = v_times[:n_val] + [max(e for _, e in blocks)]
    failed = [i for i in range(n_val)
              if v_error[i][0] > VALIDATION_ACCURACY_AVG_THRESHOLD
              or v_error[i][1] > VALIDATION_ACCURACY_MAX_THRESHOLD]

    return (100 * overlap(blinks) / recorded, len(failed), n_val,
            100 * overlap([(bounds[i], bounds[i + 1]) for i in failed]) / recorded)

def valid_trials(pid, sess, trials):
    """Per-cell trial counts and the PAS distribution behind them, for one session.

    Returns ({awareness: (valid, post-intact)}, {pas: n}). Both awareness cells of the
    session are always present, at (0, 0) if the participant never produced that response.
    The PAS counts are of every trial left after the block exclusions, PAS 1 included, so
    the responses the analysis discards stay visible next to the ones it keeps.

    The valid counts apply exactly the filters apply_behavioral_filters_and_save applies:
    the participant's excluded blocks, PAS 1 dropped, an unanswered PAS or pleasantness
    answer dropped, a null PAS dropped (polars' is_in propagates the null, so those rows
    are filtered out there too), and the trial before a broken fixation.
    """
    # EXCLUDE_BLOCKS numbers are the session's running order (1-4 Experiment, then the
    # Extra blocks), while BlockNum restarts at 1 in the Extra block. Translate to the
    # (block_type, BlockNum) pair each number names, exactly as the pipeline does, or a
    # bare number would match one block of each type.
    bad_blocks = {(("Experiment", n) if n <= N_EXPERIMENT_BLOCKS
                   else ("Extra", n - N_EXPERIMENT_BLOCKS))
                  for n in EXCL_BLOCK.get(pid, {}).get(sess, [])}
    prefix = "conscious" if sess == "C" else "unconscious"

    cells = {f"{prefix}_aware": (0, 0), f"{prefix}_unaware": (0, 0)}
    pas_counts = Counter()
    for t in trials:
        if (t["block_type"], t["block"]) in bad_blocks:
            continue
        pas = t["pas"]
        # A PAS the .mat left empty reaches polars as a null, and its is_in mask filters
        # nulls out; a PAS recorded as NaN is a real float that the mask keeps, and the
        # pipeline's when/otherwise then labels it aware. Both are reproduced here rather
        # than merged, even though every NaN in this dataset is an unanswered trial and so
        # is dropped a line later anyway.
        missing = pas is None or (isinstance(pas, float) and np.isnan(pas))
        pas_counts["na" if missing else int(pas)] += 1
        # The pilot never recorded pleasantness, so a None there is "not asked", not
        # "not answered" - the pipeline skips the rule on that dataset for the same reason.
        pls_unanswered = t["answered_pls"] is not None and not t["answered_pls"]
        if (pas is None or pas == 1 or not t["answered"] or pls_unanswered or t["pre_break"]):
            continue
        awareness = f"{prefix}_{'unaware' if pas == 0 else 'aware'}"
        total, intact = cells[awareness]
        cells[awareness] = (total + 1, intact + (1 if t["intact"] == 1 else 0))
    return cells, pas_counts

#%% Pipeline outputs: the rest of the funnel, when they exist
def load_pipeline():
    """Per-cell viewing counts from the pipeline's outputs, or None if it has not run.

    Returns (fixated, scored, final, fix_kept, stale) where the first three map a
    (pid, session, awareness) cell to a viewing count at successive stages, fix_kept maps
    (pid, session) to the % of detected fixations Stage 1 kept, and stale lists outputs
    older than their own inputs.
    """
    if not (FIX_PATH.exists() and CROSS_PKL.exists() and CROSS_CSV.exists()):
        return None

    # Viewings that still had a usable Mooney fixation after Stage 1 filtering
    fix = pd.read_parquet(FIX_PATH, columns=["participant", "session", "awareness",
                                             "image_type", "trial_number"])
    fixated = (fix[fix.image_type == "mooney_post_intact"]
               .groupby(["participant", "session", "awareness"])
               .trial_number.nunique().to_dict())

    # Viewings the cross-phase step actually scored, counted BEFORE the cell threshold.
    # This reproduces that threshold's own count: the long format is one row per subject
    # entry per reference map, and the filter counts the non-NaN ones per cell.
    with open(CROSS_PKL, "rb") as fh:
        cache = pickle.load(fh)
    cross = cache["data"] if isinstance(cache, dict) else cache
    scored = Counter()
    for entry in cross["image"]:
        session = "C" if entry["condition"] == "C" else "U"
        for subj in entry.get("subject", []):
            if pd.notna(subj.get("NSS_intact")):
                scored[(subj["ParticipantID"].split("_t")[0], session, entry["awareness"])] += 1

    # What actually reached the file the model is fitted on
    csv = pd.read_csv(CROSS_CSV, usecols=["Participant", "Session", "Awareness", "ReferenceMap", "NSS"],
                      dtype={"Participant": str})
    csv = csv[(csv.ReferenceMap == "Intact") & csv.NSS.notna()]
    final = csv.groupby(["Participant", "Session", "Awareness"]).size().to_dict()

    # Stage-1 event survival per session, for context on the gaze-loss column
    fix_kept, qc_path = {}, Path(data_quality_folder, "blink_spatial_filtering.csv")
    if qc_path.exists():
        qc = pd.read_csv(qc_path, dtype={"participant_id": str, "session_id": str})
        fix_kept = {(r.participant_id, r.session_id): 100 * r.fix_final / max(r.fix_initial, 1)
                    for r in qc.itertuples()}

    # An output older than its own input describes an older run. The cross-phase pickle is
    # deliberately not checked this way: NSS.py reuses it whenever its meta block still
    # matches, so it is legitimately older than the parquet. What catches a pickle that no
    # longer fits the data is the funnel's own consistency check, further down.
    cleaned = [f.stat().st_mtime for f in Path(EVENTS_CLEANED_DIR).glob("s_*.csv")]
    steps = [("parquet", max(cleaned, default=0), FIX_PATH),
             ("results CSV", FIX_PATH.stat().st_mtime, CROSS_CSV)]
    stale = [name for name, input_time, output in steps if output.stat().st_mtime < input_time]

    return fixated, scored, final, fix_kept, stale

#%% Walk the drop points, in the order the pipeline applies them
def diagnose(pid, sess, quality, pipeline):
    """Return (rows, pas_counts, alpha) for one session: one row per awareness cell.

    A row is (awareness, reason, funnel) with the funnel (valid trials, post-intact trials,
    fixated, scored, final), and the reason is the FIRST point that removed the cell. Both
    awareness cells are always returned, so a cell holding no trials is reported rather than
    silently absent; only an unreadable behavioural file leaves nothing to report per cell.

    A session ruled out upstream (manual exclusion, failed calibration) still reports its
    raw trial counts — that is what shows whether the exclusion is borne out by the data —
    but its pipeline columns stay blank, since the pipeline never processed it.
    """
    trials, alpha = read_trials(BEH_DIR / f"expdata_{sess}_{pid}.mat")
    if trials is None:
        return [("(all)", "no behavioural file", (None,) * 5)], Counter(), (None, False)

    cells, pas_counts = valid_trials(pid, sess, trials)
    _, n_failed, n_val, _ = quality

    # Reasons that rule out the whole session, before any cell of it is looked at
    session_reason = ("manual: subject excluded" if pid in EXCL_SUBJ else
                      "manual: session excluded" if sess in EXCL_SESS.get(pid, []) else
                      # no good interval survives -> Stage 1 keeps no samples
                      "every calibration failed" if n_val and n_failed == n_val else None)

    rows = []
    for awareness, (n_valid, n_intact) in sorted(cells.items()):
        raw_only = (n_valid, n_intact, None, None, None)
        # PAS 0 feeds the unaware cell, PAS 2 and 3 the aware one, so an empty cell can be
        # told apart: the participant never reported that level, or they did and every such
        # trial was dropped for an unanswered PAS or a repeated fixation.
        pas_for_cell = (pas_counts[0] if awareness.endswith("_unaware")
                        else pas_counts[2] + pas_counts[3])
        if session_reason:
            rows.append((awareness, session_reason, raw_only))
        elif not sum(pas_counts.values()):
            rows.append((awareness, NO_BLOCKS, raw_only))     # every block of it excluded
        elif pas_for_cell == 0:
            rows.append((awareness, NO_PAS, raw_only))        # never reported this level
        elif n_valid == 0:
            rows.append((awareness, ALL_UNUSED, raw_only))    # reported it, all trials dropped
        elif pipeline is None:
            # Raw data alone: certain only when the cell can never reach the threshold
            rows.append((awareness, THIN_TRIAL if n_intact < MIN_VIEWINGS else ELIGIBLE, raw_only))
        else:
            fixated, scored, final, _, _ = pipeline
            key = (pid, sess, awareness)
            n_fixated, n_scored, n_final = fixated.get(key, 0), scored.get(key, 0), final.get(key, 0)

            if n_intact < MIN_VIEWINGS:      reason = THIN_TRIAL   # never had the trials
            elif n_fixated == 0:             reason = NO_FIX       # every viewing lost its fixations
            elif n_fixated < MIN_VIEWINGS:   reason = THIN_FIX     # gaze loss took it below threshold
            elif n_scored == 0:              reason = NO_IMAGE     # its images had too few subjects
            elif n_final == 0:               reason = THIN_CELL    # scored, but the cell filter cut it
            else:                            reason = INCLUDED
            rows.append((awareness, reason, (n_valid, n_intact, n_fixated, n_scored, n_final)))
    return rows, pas_counts, alpha

def num(value):
    """Table cells hold text, and a stage that never ran shows as blank rather than 0."""
    return "" if value is None else f"{value}"

pipeline = load_pipeline()

# Filename grammar s_<SESSION>_<PID>.asc, as NSSExporter.py parses it. The session letter
# decides the conscious/unconscious prefix throughout the pipeline, so an unknown one would
# mislabel every awareness cell of that session rather than fail.
FILENAME = re.compile(r"s_([A-Za-z]+)_(\d+)\.asc$")
raw_files = sorted(RAW_DIR.glob("s_*.asc"))
matches = {f: FILENAME.match(f.name) for f in raw_files}
for f, m in matches.items():
    if not m:
        print(f"  ! skipping {f.name}: not s_<SESSION>_<PID>.asc")
sessions = sorted({(m.group(2), m.group(1).upper()) for m in matches.values() if m})
unknown = {sess for _, sess in sessions} - {"C", "U"}
if unknown:
    raise SystemExit(f"session letters {sorted(unknown)} are neither C nor U; awareness "
                     "labels and the reference-map split assume those two.")
if not sessions:
    raise SystemExit(f"no s_<SESSION>_<PID>.asc recordings under {RAW_DIR.resolve()}.")
print(f"Recordings   {RAW_DIR.resolve()}\nBehavioural  {BEH_DIR.resolve()}\n"
      f"Results      {CROSS_DIR if pipeline else '(none: reporting raw eligibility only)'}\n"
      f"Figure       {OUT_PNG.resolve()}\n"
      f"\nAuditing {len(sessions)} recordings (TRIAL_SET={TRIAL_SET}, MOONEY_SPLIT={MOONEY_SPLIT}, "
      f"BLINK_MODE={BLINK_MODE} -> a cell needs {MIN_VIEWINGS} trials)...")

table, cal_loss, fitted_alpha = [], {}, {}
for pid, sess in sessions:
    quality = scan_asc(RAW_DIR / f"s_{sess}_{pid}.asc")
    blink_pct, n_failed, n_val, cal_pct = quality
    cal_loss[f"{pid}{sess}"] = cal_pct   # the column reports the count; this is the cost
    kept = (pipeline[3] if pipeline else {}).get((pid, sess))
    note = "blocks " + ",".join(map(str, EXCL_BLOCK[pid][sess])) + " excl." \
           if sess in EXCL_BLOCK.get(pid, {}) else ""   # from Settings.EXCLUDE_BLOCKS

    rows, pas_counts, (alpha, alpha_passed) = diagnose(pid, sess, quality, pipeline)
    fitted_alpha[f"{pid}{sess}"] = (alpha, alpha_passed)
    # The session's answers as fractions in scale order, for the stacked bar in the PAS
    # column - one participant's bar from ResponseDistributions.py, laid on its side. A bar
    # rather than four printed percentages because what matters here is a proportion, and
    # whether PAS 0 or PAS 2/3 is thin is a length the eye reads without arithmetic.
    n_pas = sum(pas_counts.values())
    pas_shares = [pas_counts[k] / n_pas for k in PAS_LEVELS] if n_pas else None
    for first, (awareness, reason, funnel) in enumerate(rows):
        n_valid, n_intact, n_fixated, n_scored, n_final = funnel
        # Verdict is status and reason in one column: every OUT row names the step that
        # removed it, so one glance down this column is the whole audit. Any block exclusion
        # rides along in brackets rather than in a column of its own, which was empty for
        # all but the few sessions that have one.
        verdict = ("IN" if reason == INCLUDED else
                   "ELIGIBLE (pipeline not run)" if reason == ELIGIBLE else f"OUT - {reason}")
        verdict += f"   [{note}]" if note else ""
        # Kept and discarded of the same denominator (Eligible), so the two numbers on a row
        # always add up and "what do I lose here" needs no arithmetic.
        lost = None if n_final is None else n_intact - n_final
        # Session-level values are carried on the block's first row only; the figure then
        # writes them once, centred across the block, so a repeated number never reads as a
        # second, independent measurement.
        session_cell = lambda text: text if not first else ""
        table.append({
            "PID": session_cell(pid), "Ses": session_cell(sess),
            "Cell": awareness.replace("conscious_", "C-").replace("unC-", "U-"),
            "Eligible": num(n_intact), "Analysed": num(n_final),
            "Discarded": "" if lost is None else
                         f"{lost}" + (f" ({100 * lost / n_intact:.0f}%)" if n_intact else ""),
            "Verdict": verdict,
            # The image alpha this session was run at, from its own threshold fitting: the
            # stimulus strength every count on the row was produced under, so a thin cell
            # can be read against how visible the Mooney actually was for that participant.
            "Alpha": session_cell("" if alpha is None else
                                  f"{alpha:.2f}" + ("" if alpha_passed else " !")),
            "PAS": "",                  # drawn as a bar, from _pas below
            "Blink %": session_cell(f"{blink_pct:.0f}%"),
            "Gaze kept": session_cell("" if kept is None else f"{kept:.0f}%"),
            "Cal fail": session_cell(f"{n_failed}/{n_val}"),
            "_pas": pas_shares if not first else None,
            "_colour": REASONS[reason], "_funnel": funnel, "_reason": reason,
            "_kept": reason in (INCLUDED, ELIGIBLE), "_pid": pid, "_ses": sess,
            "_cell": awareness, "_lost": lost, "_eligible": n_intact,
        })

counts = Counter(r["_reason"] for r in table)

# The funnel can only ever shrink: a cell cannot score more trials than survived Stage 1,
# nor keep more rows than it scored. An inversion means two outputs describe different runs,
# which no mtime comparison would catch, so this is the real staleness test. The middle two
# steps are no longer columns of the table, but they are still the check that catches this.
inconsistent = [f"{r['_pid']}{r['_ses']} {r['_cell']}" for r in table
                if None not in r["_funnel"]
                and not r["_funnel"][1] >= r["_funnel"][2] >= r["_funnel"][3] >= r["_funnel"][4]]
warnings = ([f"{name} older than its input" for name in pipeline[4]] if pipeline else [])
warnings += [f"funnel inconsistent in {len(inconsistent)} cells: {', '.join(inconsistent[:4])}"] if inconsistent else []

#%% Report: terminal summary + the colour-coded table
n_in = counts[INCLUDED] + counts[ELIGIBLE]
print(f"\n{len(table)} participant x session x awareness cells ({n_in} kept, {len(table) - n_in} filtered out)")
for reason, n in counts.most_common():
    print(f"   {n:>3}  {reason}")

# The headline the table is built around, in trials rather than cells: how much data the
# behavioural filters leave, and how much of that the rest of the pipeline does not use.
eligible = sum(r["_eligible"] for r in table)
lost = sum(r["_lost"] for r in table if r["_lost"] is not None)
if pipeline:
    print(f"\n{eligible} eligible post-intact trials -> {eligible - lost} analysed "
          f"({100 * lost / max(eligible, 1):.0f}% discarded)")
else:
    print(f"\n{eligible} eligible post-intact trials (run the pipeline to see how many are analysed)")

# The same cells counted per awareness group: the sample each one gets, and how many
# sessions keep both of theirs (the pairs any within-session comparison can use). A row
# whose session has no behavioural file names no awareness cell, so it appears in neither.
by_session = {}
for r in table:
    if "_" in r["_cell"]:
        by_session.setdefault((r["_pid"], r["_ses"]), {})[r["_cell"]] = r["_kept"]
print("\nparticipants with a usable cell, per awareness group:")
for awareness in sorted({a for cells in by_session.values() for a in cells}):
    has = [cells for cells in by_session.values() if awareness in cells]
    print(f"   {sum(cells[awareness] for cells in has):>3} / {len(has):<3} {awareness}")
for letter in sorted({sess for _, sess in by_session}):
    both = [cells for (_, sess), cells in by_session.items()
            if sess == letter and len(cells) > 1 and all(cells.values())]
    print(f"   {len(both):>3}     both cells of their {letter} session")

# How much recording each failed validation actually costs. The table gives the count
# alone, which is the part that is comparable across sessions; the share of the recording
# it invalidates only matters for the few sessions where it is not zero.
# What stimulus strength the sample was actually run at: the fitting targets one criterion,
# so a spread here is a spread in how visible the Mooney was, not a spread in performance.
levels = Counter(f"{a:.2f}" for a, _ in fitted_alpha.values() if a is not None)
print("\nalpha after threshold fitting: " +
      ", ".join(f"{alpha} x{n}" for alpha, n in sorted(levels.items())))
never = [name for name, (alpha, ok) in sorted(fitted_alpha.items()) if alpha is not None and not ok]
if never:
    print(f"  ! never reached criterion: {', '.join(never)}")

costly = {name: pct for name, pct in cal_loss.items() if pct >= 0.5}
print("\nrecording lost to failed calibration: " +
      (", ".join(f"{name} {pct:.0f}%" for name, pct in sorted(costly.items())) if costly
       else "none above 0.5% in any session"))

for warning in warnings:
    print(f"  ! STALE: {warning} - rerun the pipeline.")
if pipeline and not warnings:
    print("  funnel consistent across all cells: parquet, pickle and results CSV agree.")

cols = [c for c in table[0] if not c.startswith("_")]
# Widths sum to 1, so the table fills the axes: identity, the three numbers the audit is
# about, the verdict that explains them, then the session-level context.
WIDTHS = dict(zip(cols, [.032, .024, .075, .062, .062, .070, .327, .051, .115, .062, .065, .055]))
# Second header line: what the column counts, so a number needs no legend to be read.
SUBHEADS = {"Cell": "awareness", "Eligible": "trials", "Analysed": "trials",
            "Discarded": "trials", "Verdict": "what removed the cell",
            "Alpha": "threshold", "PAS": "response mix", "Blink %": "recording",
            "Gaze kept": "Stage 1", "Cal fail": "of total"}
# Columns describing the session rather than the awareness cell. They are left out of the
# table's own text and drawn afterwards, centred across the participant's rows, which is
# what gives each participant one merged-looking block instead of two separate rows.
SESSION_COLS = ("PID", "Ses", "Alpha", "PAS", "Blink %", "Gaze kept", "Cal fail")
# Where each participant's block starts, and which block every row belongs to.
starts = [i for i, r in enumerate(table)
          if i == 0 or (r["_pid"], r["_ses"]) != (table[i - 1]["_pid"], table[i - 1]["_ses"])]
block_of = {i: sum(start <= i for start in starts) - 1 for i in range(len(table))}
# The session columns are banded per participant instead of washed with the verdict colour:
# the value describes the session, not the cell, and a block of one flat colour is what lets
# a centred value sit across both rows without a cell edge cutting through it.
BANDS = ("#ffffff", "#eeeeee")
fig, ax = plt.subplots(figsize=(14, 0.26 * len(table) + 1.6))
ax.axis("off")

# The header holds two lines and so is twice as tall as a data row; both of its lines are
# drawn by hand (below), since a cell renders one string in one style.
UNIT = 1 / (len(table) + 2)
tbl = ax.table(cellText=[["" if c in SESSION_COLS else r[c] for c in cols] for r in table],
               colLabels=[""] * len(cols), cellLoc="left", loc="upper center",
               colWidths=[WIDTHS[c] for c in cols])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

for (row, col), cell in tbl.get_celld().items():
    cell.set_height(2 * UNIT if row == 0 else UNIT)  # heights sum to 1: no dead space
    cell.set_edgecolor("white")
    if row == 0:
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", weight="bold")
        continue
    if cols[col] in SESSION_COLS:
        band = BANDS[block_of[row - 1] % 2]
        cell.set_facecolor(band)
        cell.set_edgecolor(band)      # no divider inside a block, so the value can span it
        continue
    # The verdict carries the colour at full strength; the rest of the row gets a faint
    # wash of it so the eye can follow a row without the page turning into a colour chart.
    cell.set_facecolor(table[row - 1]["_colour"])
    cell.set_alpha(1.0 if cols[col] == "Verdict" else 0.35)

# A participant's two awareness cells are one entry, not two unrelated rows: a rule closes
# off each block, and the session-level values are written once, vertically centred across
# the block, the way a merged cell would read. Both are drawn on the axes rather than as
# cell properties - a cell edge would inherit the two different alphas above, and a cell's
# own text is repositioned on every draw, so it cannot be moved off its row.
# The geometry comes from the drawn cells, in axes coordinates (which is what a table cell
# reports, and what both artists below must therefore be given): the table is NOT flush with
# the axes - it is placed with a small inset - so the nominal row height puts a line a whole
# row out of place, and drawing in data coordinates would rescale the axes under the table.
fig.canvas.draw()
def cell_text(cell, text, **kwargs):
    """Text at a cell's own left padding, so it lines up with the table's own cells."""
    return ax.text(cell.get_x() + cell.PAD * cell.get_width(), kwargs.pop("y"), text,
                   transform=ax.transAxes, ha="left", zorder=6, **kwargs)

# Header: the name, then in smaller grey what the column holds, which is what stops a bare
# count from having to be looked up in the footnote.
head = tbl[0, 0]
for col, name in enumerate(cols):
    cell = tbl[0, col]
    cell_text(cell, name, y=head.get_y() + 0.62 * head.get_height(),
              va="center", color="white", weight="bold", fontsize=8)
    cell_text(cell, SUBHEADS.get(name, ""), y=head.get_y() + 0.28 * head.get_height(),
              va="center", color="#b8b8b8", fontsize=6.5)

last_col = tbl[1, len(cols) - 1]
x0, x1 = tbl[1, 0].get_x(), last_col.get_x() + last_col.get_width()
for block, start in enumerate(starts):
    end = starts[block + 1] - 1 if block + 1 < len(starts) else len(table) - 1
    top = tbl[start + 1, 0].get_y() + tbl[start + 1, 0].get_height()
    bottom = tbl[end + 1, 0].get_y()
    if start:                   # the first block needs no rule: the header row closes it
        ax.plot([x0, x1], [top, top], transform=ax.transAxes,
                color="#444444", lw=1.0, clip_on=False, zorder=5)
    for col, name in enumerate(cols):
        if name not in SESSION_COLS:
            continue
        cell = tbl[start + 1, col]
        if name != "PAS":
            cell_text(cell, table[start][name], y=(top + bottom) / 2, va="center", fontsize=8)
        elif table[start]["_pas"]:
            # The session's answers as one 100% bar, PAS 0 at the left so the segment the
            # unaware cell is built from starts at a fixed edge and its length is comparable
            # down the column. A session with no answers at all gets no bar rather than a
            # full-width grey one, which would read as "all unanswered".
            pad = cell.PAD * cell.get_width()
            x0_bar, width = cell.get_x() + pad, cell.get_width() - 2 * pad
            height = min(0.55 * (top - bottom), 0.6 * UNIT)
            y_bar, x = (top + bottom) / 2 - height / 2, x0_bar
            for share, colour in zip(table[start]["_pas"], PAS_COLOURS):
                ax.add_patch(plt.Rectangle((x, y_bar), share * width, height,
                                           transform=ax.transAxes, facecolor=colour,
                                           edgecolor="white", lw=0.4, zorder=6, clip_on=False))
                x += share * width
            # A frame, because PAS 0 is the palette's lightest colour and would otherwise
            # fade into the band: the bar has to read as 100% of the session either way.
            ax.add_patch(plt.Rectangle((x0_bar, y_bar), width, height, transform=ax.transAxes,
                                       fill=False, edgecolor="#9a9a9a", lw=0.5,
                                       zorder=7, clip_on=False))

swatch = lambda colour: plt.Rectangle((0, 0), 1, 1, fc=colour)
legend = ax.legend([swatch(REASONS[r]) for r, _ in counts.most_common()],
                   [f"{r}  (n={n})" for r, n in counts.most_common()],
                   loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                   frameon=False, fontsize=8)
legend.set_title(
    "One shaded block per participant, one row per awareness cell. PID, Ses, Alpha, PAS and "
    "the recording context describe the session, so they are written once, across the block."
    "\n\n"
    "Eligible, Analysed and Discarded all count POST-INTACT MOONEY TRIALS - one per viewing, "
    "so an image seen again in the Extra block counts twice:\n"
    "    Eligible    trials left after the behavioural exclusions (excluded sessions and "
    "blocks, PAS 1, an unanswered PAS or pleasantness, the trial before a broken fixation).\n"
    "    Analysed    of those, the ones whose Mooney fixations were scored against the "
    "disambiguator maps and reached the fitted dataset; blank = never processed.\n"
    "    Discarded   the difference: trials lost to Stage-1 gaze filtering (blinks, failed "
    f"calibration, fixations off the image or within {CENTER_RADIUS_DG}\u00b0 of centre), or to\n"
    f"                the {MIN_VIEWINGS_PER_IMAGE_CROSS}-viewings-per-image and "
    f"{MIN_IMAGES_PER_CELL_CROSS}-scores-per-cell thresholds.\n\n"
    "Verdict = the FIRST step that removed a cell, with any block exclusion in brackets.\n"
    "Alpha = the image alpha this session's threshold fitting settled on, and every trial "
    "above was then run at (\"!\" = the fitting never reached criterion).\n"
    "PAS bar = the session's answers after the block exclusions, PAS 0 at the left (PAS 1 is "
    "always dropped; PAS 0 feeds the unaware cell, PAS 2 and 3 the aware one).\n"
    "Recording context: Blink % of the recording | Gaze kept = fixations surviving Stage 1 | "
    "Cal fail = validations missing the accuracy threshold.",
    prop={"size": 7.5})

# The PAS key goes under the verdict key rather than beside it: two legends at one anchor
# overlap, so this one is placed at the measured bottom of the first.
ax.add_artist(legend)
fig.canvas.draw()
below = legend.get_window_extent().transformed(ax.transAxes.inverted()).y0
ax.legend([swatch(c) for c in PAS_COLOURS], PAS_LABELS, loc="upper center",
          bbox_to_anchor=(0.5, below - 0.004), ncol=5, frameon=False, fontsize=8)
ax.set_title(f"Who the NSS analysis keeps — {DATASET}, TRIAL_SET={TRIAL_SET}, "
             f"MOONEY_SPLIT={MOONEY_SPLIT}, BLINK_MODE={BLINK_MODE}"
             + ("   [RAW ONLY: pipeline not run]" if pipeline is None else
                f"   [STALE: {'; '.join(warnings)}]" if warnings else ""),
             fontsize=11, weight="bold")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT_PNG}")

# %%
