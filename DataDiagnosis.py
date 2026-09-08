"""Per-cell inclusion audit: exactly whose data is filtered out of the NSS analysis, and where.

Data leaves this analysis at six different points — the manual exclusions, calibration,
the behavioural trial filters, Stage-1 gaze filtering, the per-image subject threshold and
the per-cell image threshold — spread over Settings.py, three scripts and two stages. A
participant missing from the results says nothing about which of those removed them. This
follows one row per participant x session x awareness cell (the unit the NSS cell filter
acts on) through all six as a funnel:

    valid trials  ->  viewings with usable fixations  ->  scored viewings  ->  final data

so a cell's row shows both that it was dropped and how much data it had left at each step.
The Reason column names the FIRST point that removed it, colour-coded so the dominant
cause across the sample is visible at a glance.

Every session lists BOTH of its awareness cells, including the ones that hold no trials at
all and the ones of a manually excluded session, so a missing cell is always visible as a
row with a reason rather than as an absence. That is also what makes the PAS column worth
reading: it is the session's full response distribution, PAS 1 included, so a cell that is
thin because the participant rarely reported that awareness level says so on its own row.

The raw files (.asc + .mat) carry the funnel's first column and every upstream reason, so
the script runs standalone. The last three columns come from the pipeline's outputs; when
those are absent it still runs, marks the cells ELIGIBLE instead of IN, and says so — an
eligibility ceiling rather than a membership list, since raw data cannot see gaze loss.
Outputs older than their own inputs are reported as stale rather than trusted.

Strictly read-only: it recomputes nothing, creates no folders, and writes only the figure.

The data need not sit in the repository, and a second dataset is audited without touching
anything else: DATASET_FOLDER and APPLY_EXCLUSIONS below are the two switches, set once and
then run the file. Settings.py's EXCLUDE_* lists name participants of the dataset it is set
up for, and participant numbers repeat across datasets, so another one is audited with
APPLY_EXCLUSIONS off or it removes whoever happens to share an ID. Every switch is also a
command-line option, for a run that leaves the file alone:

    python3 DataDiagnosis.py --data ~/Desktop/PilotTwo --no-exclusions --out ~/Desktop/audit.png
    python3 DataDiagnosis.py --help          # every path it can be told about

Reads:  <data>/raw/s_<SESSION>_<PID>.asc                          (blinks, calibration, recording spans)
        <data>/behavioural/expdata_<SESSION>_<PID>.mat            (trials, PAS, blocks)
        Settings.py                                               (exclusions and thresholds)
        default <data> is data/my_dataset/
        optional, for the exact verdict:
        DataQualityChecks/blink_spatial_filtering.csv             (CompleteRun.py)
        data/NSS_all_fixations_clean.parquet                      (NSSExporter.py)
        analysesresults/NSS_<mode>/NSS_crossphase_descriptives.pkl (NSS.py)
        analysesresults/NSS_<mode>/NSS_CrossPhase_LongFormat.csv   (NSS.py)
Writes: Figures/DataDiagnosis.png                                  (or --out)
"""

#%% Imports and configuration
from pathlib import Path
from collections import Counter
import argparse
import mmap
import pickle
import re
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # project root, for the imports below
from Settings import (RAW_DATA_DIR, BEHAVIOURAL_DIR, EVENTS_CLEANED_DIR, FIX_FILE,
                      data_quality_folder, SECTION_TO_BLOCK, TRIAL_SET, MOONEY_SPLIT,
                      INTERPOLATE_BLINKS, EXCLUDE_SUBJECTS, EXCLUDE_SESSIONS, EXCLUDE_BLOCKS,
                      VALIDATION_ACCURACY_AVG_THRESHOLD, VALIDATION_ACCURACY_MAX_THRESHOLD,
                      MIN_SUBJ_PER_IMAGE_CROSS, MIN_IMAGES_PER_CELL_CROSS)
from Scripts.Analysis.NSS.NSSPaths import TRIAL_SET_SUFFIX, BLINK_SUFFIX

# The results folder for the modes in Settings.py. Built from NSSPaths' own suffix maps
# rather than NSSPaths.select(), which would create the folder as a side effect.
BLINK_MODE   = "interp" if INTERPOLATE_BLINKS else "filter"
DEFAULT_CROSS = Path(f"analysesresults/NSS_{MOONEY_SPLIT}"
                     f"{TRIAL_SET_SUFFIX[TRIAL_SET]}{BLINK_SUFFIX[BLINK_MODE]}")

#%% Run switches: what a plain "Run" on this file uses (the command line overrides both)
# Set for the dataset being audited, then run the file; nothing else in the repo moves.
DATASET_FOLDER   = "data_rep/my_dataset"  # None -> Settings.py's own data/my_dataset
APPLY_EXCLUSIONS = False                  # False -> judge every cell on its own data

#%% Where the data is: the switches above, then the command line, then Settings.py
def parse_args():
    """Resolve every path this script reads and writes.

    The data does not have to live in the repository: point --data at a dataset folder
    anywhere and its raw/ and behavioural/ subfolders are picked up (or the folder itself,
    if it holds the recordings directly). Each path can also be given outright. Everything
    left unset falls back to the repo layout in Settings.py, so a bare run is unchanged.
    """
    parser = argparse.ArgumentParser(
        description="Audit which participants the NSS analysis keeps, and what removes the rest.",
        epilog="examples:\n"
               "  python3 DataDiagnosis.py\n"
               "  python3 DataDiagnosis.py --data ~/Desktop/PilotTwo\n"
               "  python3 DataDiagnosis.py --data /Volumes/Backup/study --out ~/Desktop/audit.png",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--data", metavar="DIR", default=DATASET_FOLDER,
        help="dataset folder holding raw/ and behavioural/ "
             f"(default: {DATASET_FOLDER or Path(RAW_DATA_DIR).parent})")
    add("--raw", metavar="DIR", help="the .asc recordings, if they are not in <data>/raw")
    add("--behavioural", metavar="DIR", help="the .mat files, if they are not in <data>/behavioural")
    add("--parquet", metavar="FILE", help=f"fixations parquet, for the exact verdict (default: {FIX_FILE})")
    add("--results", metavar="DIR", help=f"NSS results folder, for the exact verdict (default: {DEFAULT_CROSS})")
    add("--out", metavar="FILE", default="Figures/DataDiagnosis.png", help="figure to write (default: %(default)s)")
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

# One colour per outcome, in cascade order. The key is also the text in the Reason column,
# so the legend, the cell colours and the terminal summary can never drift apart.
INCLUDED   = "included"
ELIGIBLE   = "eligible (pipeline not run)"
THIN_TRIAL = f"under {MIN_VIEWINGS} valid trials"
NO_FIX     = "all fixations lost in Stage 1"
THIN_FIX   = f"under {MIN_VIEWINGS} viewings survived Stage 1"
NO_BLOCKS  = "no trials left after block exclusions"
NO_PAS     = "no trials with this PAS response"
ALL_UNUSED = "every trial of this PAS unanswered or repeated"
NO_IMAGE   = f"no image reached {MIN_SUBJ_PER_IMAGE_CROSS} subjects"
THIN_CELL  = f"under {MIN_IMAGES_PER_CELL_CROSS} valid scores"
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
    """Return one dict per real trial of a behavioural .mat, or None if it cannot be read.

    Mirrors load_behavioural_from_mat (section walk, ghost-trial skip) but reads the fields
    directly: the audit needs IsIntactDisambiguation, which is not in Settings.MAT_FIELD_MAP,
    and no other consumer of that map should have to grow a column for this script.
    """
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        expdata = next(v for k, v in mat.items() if k.lower() == "expdata")
    except Exception:
        return None

    def field(trial, name):
        """One scalar field, with MATLAB's empty placeholders normalised to None."""
        value = getattr(trial, name, None)
        return None if isinstance(value, np.ndarray) and value.size == 0 else value

    trials = []
    for section, block in SECTION_TO_BLOCK.items():
        if block not in WANTED_BLOCKS or not hasattr(expdata, section):
            continue
        struct = getattr(expdata, section)
        for trial in (struct if isinstance(struct, np.ndarray) else [struct]):
            if field(trial, "TrialNum") is None:      # ghost trial: an empty placeholder row
                continue
            trials.append({k: field(trial, src) for k, src in
                           [("block", "BlockNum"), ("pas", "response_PAS_Q"),
                            ("answered", "did_answer_PAS_Q"),
                            ("fixfail", "NumRepetitionFixationFail"),
                            ("intact", "IsIntactDisambiguation")]})

    # Without the intact flag every trial would look scrambled and every cell would be
    # reported as too thin, which reads like a result rather than a missing field.
    if trials and all(t["intact"] is None for t in trials):
        raise SystemExit(f"{Path(mat_path).name}: no IsIntactDisambiguation field in the "
                         "expdata trials, so post-intact viewings cannot be counted.")
    return trials

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
    the participant's excluded blocks, PAS 1 dropped, an unanswered PAS dropped, a null PAS
    dropped (polars' is_in propagates the null, so those rows are filtered out there too),
    and any trial repeated for failing fixation.
    """
    bad_blocks = EXCL_BLOCK.get(pid, {}).get(sess, [])
    prefix = "conscious" if sess == "C" else "unconscious"

    cells = {f"{prefix}_aware": (0, 0), f"{prefix}_unaware": (0, 0)}
    pas_counts = Counter()
    for t in trials:
        if t["block"] in bad_blocks:
            continue
        pas = t["pas"]
        # A PAS the .mat left empty reaches polars as a null, and its is_in mask filters
        # nulls out; a PAS recorded as NaN is a real float that the mask keeps, and the
        # pipeline's when/otherwise then labels it aware. Both are reproduced here rather
        # than merged, even though every NaN in this dataset is an unanswered trial and so
        # is dropped a line later anyway.
        missing = pas is None or (isinstance(pas, float) and np.isnan(pas))
        pas_counts["na" if missing else int(pas)] += 1
        if (pas is None or pas == 1 or not t["answered"] or (t["fixfail"] or 0) > 0):
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
    """Return (rows, pas_counts) for one session: one row per awareness cell.

    A row is (awareness, reason, funnel) with the funnel (valid trials, post-intact trials,
    fixated, scored, final), and the reason is the FIRST point that removed the cell. Both
    awareness cells are always returned, so a cell holding no trials is reported rather than
    silently absent; only an unreadable behavioural file leaves nothing to report per cell.

    A session ruled out upstream (manual exclusion, failed calibration) still reports its
    raw trial counts — that is what shows whether the exclusion is borne out by the data —
    but its pipeline columns stay blank, since the pipeline never processed it.
    """
    trials = read_trials(BEH_DIR / f"expdata_{sess}_{pid}.mat")
    if trials is None:
        return [("(all)", "no behavioural file", (None,) * 5)], Counter()

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
    return rows, pas_counts

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
      f"BLINK_MODE={BLINK_MODE} -> a cell needs {MIN_VIEWINGS} viewings)...")

table = []
for pid, sess in sessions:
    quality = scan_asc(RAW_DIR / f"s_{sess}_{pid}.asc")
    blink_pct, n_failed, n_val, cal_pct = quality
    kept = (pipeline[3] if pipeline else {}).get((pid, sess))
    note = "blocks " + ",".join(map(str, EXCL_BLOCK[pid][sess])) + " excluded" \
           if sess in EXCL_BLOCK.get(pid, {}) else ""   # from Settings.EXCLUDE_BLOCKS

    rows, pas_counts = diagnose(pid, sess, quality, pipeline)
    # Share of the session's trials at each rating, always in 0/1/2/3 order (rounded, so
    # the four need not total 100). Unrecorded ratings are appended only when they occur.
    n_pas = sum(pas_counts.values())
    share = lambda key: round(100 * pas_counts[key] / n_pas) if n_pas else 0
    pas = "/".join(str(share(k)) for k in (0, 1, 2, 3)) + "%"
    pas += f"  +{share('na')}%na" if share("na") else ""    # only when it rounds above 0
    for awareness, reason, funnel in rows:
        n_valid, n_intact, n_fixated, n_scored, n_final = funnel
        table.append({
            "PID": pid, "Ses": sess,
            "Awareness": awareness.replace("conscious_", "C-").replace("unC-", "U-"),
            "Status": "IN" if reason == INCLUDED else "ELIGIBLE" if reason == ELIGIBLE else "OUT",
            "Reason": "" if reason in (INCLUDED, ELIGIBLE) else reason,
            "PAS 0/1/2/3": pas,
            "Valid": num(n_valid), "Post-intact": num(n_intact),
            "Fixated": num(n_fixated), "Scored": num(n_scored), "Final": num(n_final),
            "Blink %": f"{blink_pct:.0f}%",
            "Fix kept": "" if kept is None else f"{kept:.0f}%",
            "Cal lost": f"{cal_pct:.0f}% ({n_failed}/{n_val})",
            "Note": note,
            "_colour": REASONS[reason], "_funnel": funnel,
        })

counts = Counter(r["Reason"] or (INCLUDED if r["Status"] == "IN" else ELIGIBLE) for r in table)

# The funnel can only ever shrink: a cell cannot score more viewings than survived Stage 1,
# nor keep more rows than it scored. An inversion means two outputs describe different runs,
# which no mtime comparison would catch, so this is the real staleness test.
inconsistent = [f"{r['PID']}{r['Ses']} {r['Awareness']}" for r in table
                if None not in r["_funnel"]
                and not r["_funnel"][1] >= r["_funnel"][2] >= r["_funnel"][3] >= r["_funnel"][4]]
warnings = ([f"{name} older than its input" for name in pipeline[4]] if pipeline else [])
warnings += [f"funnel inconsistent in {len(inconsistent)} cells: {', '.join(inconsistent[:4])}"] if inconsistent else []

#%% Report: terminal summary + the colour-coded funnel table
n_in = counts[INCLUDED] + counts[ELIGIBLE]
print(f"\n{len(table)} participant x session x awareness cells ({n_in} kept, {len(table) - n_in} filtered out)")
for reason, n in counts.most_common():
    print(f"   {n:>3}  {reason}")
# The same cells counted per awareness group: the sample each one gets, and how many
# sessions keep both of theirs (the pairs any within-session comparison can use). A row
# whose session has no behavioural file names no awareness cell, so it appears in neither.
by_session = {}
for r in table:
    if "-" in r["Awareness"]:
        by_session.setdefault((r["PID"], r["Ses"]), {})[r["Awareness"]] = r["Status"] != "OUT"
print("\nparticipants with a usable cell, per awareness group:")
for awareness in sorted({a for cells in by_session.values() for a in cells}):
    has = [cells for cells in by_session.values() if awareness in cells]
    print(f"   {sum(cells[awareness] for cells in has):>3} / {len(has):<3} {awareness}")
for letter in sorted({sess for _, sess in by_session}):
    both = [cells for (_, sess), cells in by_session.items()
            if sess == letter and len(cells) > 1 and all(cells.values())]
    print(f"   {len(both):>3}     both cells of their {letter} session")

for warning in warnings:
    print(f"  ! STALE: {warning} - rerun the pipeline.")
if pipeline and not warnings:
    print("  funnel consistent across all cells: parquet, pickle and results CSV agree.")

cols = [c for c in table[0] if not c.startswith("_")]
fig, ax = plt.subplots(figsize=(18, 0.24 * len(table) + 1.4))
ax.axis("off")

tbl = ax.table(cellText=[[r[c] for c in cols] for r in table], colLabels=cols,
               cellLoc="left", loc="upper center",
               colWidths=[.033, .025, .076, .056, .185, .095, .043, .056, .045, .043, .038,
                          .045, .056, .078, .126])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

for (row, col), cell in tbl.get_celld().items():
    cell.set_height(1 / (len(table) + 1))   # fill the axes exactly, so no dead space appears
    cell.set_edgecolor("white")
    if row == 0:
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", weight="bold")
    else:
        # Status and Reason carry the colour; the rest of the row gets a faint wash of it
        # so the eye can follow a row without the page turning into a colour chart.
        cell.set_facecolor(table[row - 1]["_colour"])
        cell.set_alpha(1.0 if cols[col] in ("Status", "Reason") else 0.35)

handles = [plt.Rectangle((0, 0), 1, 1, fc=REASONS[r]) for r, _ in counts.most_common()]
legend = ax.legend(handles, [f"{r}  (n={n})" for r, n in counts.most_common()],
                   loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)
legend.set_title(
    "PAS 0/1/2/3 = share of the session's trials at each rating, after the block exclusions "
    "(PAS 1 is always dropped; PAS 0 feeds the unaware cell, PAS 2 and 3 the aware one).\n"
    "Funnel, left to right: Valid = trials passing the behavioural filters | Post-intact = those "
    "with an intact disambiguator | Fixated = viewings still carrying a usable fixation |\n"
    "Scored = viewings the cross-phase step scored | Final = rows in the fitted dataset.   "
    "Context: Blink % of recording | Fix kept = fixations surviving Stage 1 | Cal lost = recording "
    "dropped for bad calibration (failed/total).", prop={"size": 7.5})
ax.set_title(f"NSS inclusion audit — TRIAL_SET={TRIAL_SET}, MOONEY_SPLIT={MOONEY_SPLIT}, "
             f"BLINK_MODE={BLINK_MODE}" + ("   [RAW ONLY: pipeline not run]" if pipeline is None else
             f"   [STALE: {'; '.join(warnings)}]" if warnings else ""),
             fontsize=11, weight="bold")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved {OUT_PNG}")

# %%
