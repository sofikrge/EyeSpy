"""Single source of truth for the whole pipeline. This is the only file to edit.

Both stages read their parameters from here and configure nothing themselves:

    Stage 1 (preprocessing)   screen geometry, thresholds, blink handling, marker
                              codes, exclusions, and the pymovements dataset
    Stage 2 (NSS analysis)    canvas size, pixels-per-degree, subject thresholds

What differs between the pilot and the replication (recording rig, exclusions) lives
in Settings_pilot.py / Settings_rep.py, applied on top of this file for the dataset
chosen at the top.

One thing to know before changing anything: DEBUG gates I/O, not just logging. With
DEBUG on, the intermediate Data.nosync/<dataset>/events/ CSVs and every QC figure under
Data_Quality_Checks/<dataset>/ get written.
"""

#%% Imports
import os
import re
import polars as pl
import matplotlib.pyplot as plt
import pymovements as pm
from pathlib import Path

#%% ============================ DATASET SELECTION ==============================
# Which dataset this run works on: "pilot" or "rep". Taken from EYESPY_DATASET if set
# (a script's DATASET toggle, or `EYESPY_DATASET=rep python3 ...`), otherwise asked in
# the terminal. The answer is written back to the environment so Stage-2 subprocesses
# inherit it instead of asking again. It picks everything that differs per dataset:
# the Data.nosync/<dataset>/ input folder, the <dataset>/ subfolder of every output folder, and
# Settings_<dataset>.py, applied at the bottom of this file on top of the values here.
# Every input and output folder gets one <dataset>/ subfolder per dataset; the data sits
# under Data.nosync/ so iCloud does not upload it, and _SUFFIX tags figure filenames.
DATASETS = ("pilot", "rep")
DATASET  = os.environ.get("EYESPY_DATASET") or input("Analyse which dataset? [pilot/rep]: ").strip()
if DATASET not in DATASETS:
    raise SystemExit(f"Unknown dataset {DATASET!r}; choose one of {DATASETS}.")
os.environ["EYESPY_DATASET"] = DATASET

DATA_ROOT = f"Data.nosync/{DATASET}"
_SUFFIX   = f"_{DATASET}"

#%% =========================== STAGE 1: PREPROCESSING ===========================

DEBUG = True

# Set per dataset in Settings_<dataset>.py, not here: the recording rig (SCREEN,
# EYE_OFFSET, IMAGE_SIZE_DEG, MASK_PPD) and the exclusions (EXCLUDE_SUBJECTS,
# EXCLUDE_SESSIONS, EXCLUDE_BLOCKS).

filename_format = {'gaze': r's_{session_id:s}_{participant_id:d}.asc'}
filename_format_schema_overrides = {'gaze': {'session_id': str, 'participant_id': str}}

time_column, time_unit = 'time', 'ms'
pixel_columns = ['x_right', 'y_right', 'x_left', 'y_left']

# Folders (each writer creates it on demand, so nothing is made just by importing)
data_quality_folder = f'Data_Quality_Checks/{DATASET}/'

# Validation thresholds
VALIDATION_ACCURACY_AVG_THRESHOLD = 1.0  # degrees
VALIDATION_ACCURACY_MAX_THRESHOLD = 1.5  # degrees

RAW_DATA_DIR = os.path.join(DATA_ROOT, 'my_dataset', 'raw') # path to raw data for manual blink parsing
EVENTS_OUT_DIR = os.path.join(DATA_ROOT, 'events')
BEHAVIOURAL_DIR = Path(f"{DATA_ROOT}/my_dataset/behavioural") # path to behavioural data
EVENTS_CLEANED_DIR = os.path.join(DATA_ROOT, 'events_cleaned')

FIX_VELOCITY_THRESHOLD = 30.0  # degrees per second
MIN_FIX_DURATION_MS = 50      # minimum fixation length

# Buffers for the blink-drop path below (INTERPOLATE_BLINKS = False) only. These are
# not the preregistered 200 ms: that is BLINK_MARGIN_MS, on the interpolation path.
BUFFER_FIX = 51   # 51ms for fixations
BUFFER_SAC = 60   # 50ms + 10ms for saccades

# --- Blink handling: two mutually exclusive modes
#   True  -> PCHIP-interpolate position across short blinks before event detection, so a
#            blink-spanning fixation survives as one fixation. This is the preregistered
#            method and what both datasets are run with.
#   False -> drop events overlapping a blink instead, using the buffers above. The older
#            method, kept as a robustness check; it is NOT the registered one.
# Rationale and sources for all three values: REFERENCES.md.
INTERPOLATE_BLINKS = True
MAX_BLINK_INTERP_MS = 150   # longer gaps are track loss, left as gaps
BLINK_MARGIN_MS = 200       # each blink is widened by this before interpolating

CENTER_RADIUS_DG = 1.5 #shaked's value

# Event markers
TRIAL_LABELS = {
    'intact': '206',          # Start of Intact Disambiguating Image
    'not_intact': '207',      # Start of Scrambled Disambiguating Image
    'disamb_end': '208',      # End of Disambiguating Image
    'mooney_steady': '210',   # Start of Mooney Image
    'mooney_end': '211'       # End of Mooney Image
}

# Regex Patterns (How to find trials and images in the ASC text)
# Only change these if the ASC file structure changes significantly
ASC_PATTERNS = {
    'trial': re.compile(r"MSG\s+(\d+)\s+TrialId_Overall\s+(\w+):\s*(\d+)"),
    'image': re.compile(r"MSG\s+(\d+)\s+ImageNumber:\s*(\d+)"),
    'msg':   re.compile(r"MSG\s+(\d+)\s+(\d+)")
}

SECTION_TO_BLOCK = {"Trials_Practice": "Practice", "Trials_Experiment": "Experiment", "Trials_Extra": "Extra"}

# MAT struct field name -> output column name (behavioural data merge)
MAT_FIELD_MAP = {
    'BlockNum': 'BlockNum',
    'ImageName': 'ImageName',
    'did_answer_PAS_Q': 'DidRespondPas',
    'did_answer_PLS_Q': 'DidRespondPls',
    'NumRepetitionFixationFail': 'NumRepetitionFixationFail',
    'response_PAS_Q': 'response_PAS_Q',
}

# How many Experiment blocks a session has before its Extra blocks begin. Only used to
# read EXCLUDE_BLOCKS (Settings_<dataset>.py); every session of both datasets records
# exactly four.
N_EXPERIMENT_BLOCKS = 4

# Plotting colours
FILTER_PALETTE = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']
PHASE_PALETTE  = ['#b3cde3', '#8c96c6', '#88419d']


#%% =========================== STAGE 2: NSS ANALYSIS ============================
# Read by NSS.py and by the Checks/ and Plots/ scripts. 

# --- Run modes: which version of the results this run reads and writes
#   MOONEY_SPLIT  "whole"  score the full 3 s Mooney presentation as one unit
#                 "halves" score each 1.5 s half (Early/Late) against the same refs
#   TRIAL_SET     "all" | "experiment" (Experiment block only) | "extra" (Extra only)
# BLINK_MODE is not set here: it is derived from INTERPOLATE_BLINKS above, so the
# analysis can never read a results folder that Stage 1 did not build.
# Output folders are suffixed to match, e.g. NSS_whole, NSS_whole_exponly_interp.
MOONEY_SPLIT = "whole" # "whole" | "halves"
TRIAL_SET    = "all" # "all" | "experiment" | "extra"

NSS_DEBUG = True   # separate from Stage 1's DEBUG: prints per-image NSS diagnostics

# Fixation-map canvas, not the screen resolution: fixations are mapped into it with
# degree 0 at its centre. Inherited from the MATLAB implementation (REFERENCES.md).
IMAGE_HEIGHT = 600
IMAGE_WIDTH  = 800

FIX_FILE = Path(f"{DATA_ROOT}/NSS_all_fixations_clean.parquet")   # written by NSSExporter.py
ANALYSES_ROOT = f"Analysis_Results/{DATASET}"   # NSSPaths.py builds its NSS_* folders in here
FIGURES_ROOT  = f"Figures/{DATASET}"           # every Plots/ script writes in here

# How much data an image needs before it is scored at all.
MIN_SUBJ_PER_IMAGE_NSS    = 2    # within-phase: minimum subjects per image
# Counts VIEWINGS, not people: one participant who saw an image in both the Experiment
# and the Extra block already satisfies a threshold of 2.
MIN_VIEWINGS_PER_IMAGE_CROSS = 2  # cross-phase: minimum Mooney viewings per image
MIN_IMAGES_PER_CELL_CROSS = 15   # cross-phase: minimum valid scores per participant cell

# Missing reference maps: "permissive" averages whatever is present,
# "matlab_strict" requires both references or scores the image NaN.
NAN_POLICY_CROSS = "permissive"

DISPERSION_DDOF = 0   # population sd, for MATLAB parity (REFERENCES.md)

# Reporting thresholds, used only by Scripts/Analysis/Checks/.
MIN_IMAGES_PER_PARTICIPANT = 15   # ImagePerParticipant.py flags anyone below this
MIN_FIX_PER_PARTICIPANT    = 20   # LeftBiasPerParticipant.py ignores thinner cells


#%% ======================== PER-DATASET VALUES ==========================
# Applied last, so Settings_<dataset>.py can also override anything above.
print(f"[Settings] EYESPY_DATASET={DATASET} -> applying Settings_{DATASET}.py")
exec(f"from Settings_{DATASET} import *")


#%% ============================== DERIVED VALUES ================================
# Built last, from the values above as the overrides left them. Nothing here is a
# parameter: every line restates one of them, so overriding SCREEN, IMAGE_SIZE_DEG or
# MASK_PPD in Settings_<dataset>.py reaches the pymovements dataset, the image bounds and the
# blur radius instead of being silently ignored.

HX, HY = IMAGE_SIZE_DEG[0] / 2, IMAGE_SIZE_DEG[1] / 2   # image half-extent, degrees
SIGMA  = MASK_PPD / 2.0                                 # fixation-map blur radius, 0.5 deg

dataset_paths = pm.DatasetPaths(
    root=f'{DATA_ROOT}/',
    raw='raw',
    preprocessed='preprocessed',
    events='events')

experiment = pm.gaze.Experiment(
    screen_width_px=SCREEN["width_px"], screen_height_px=SCREEN["height_px"],
    screen_width_cm=SCREEN["width_cm"], screen_height_cm=SCREEN["height_cm"],
    distance_cm=SCREEN["distance_cm"], origin=SCREEN["origin"],
    sampling_rate=SCREEN["sampling_rate"])

dataset_definition = pm.DatasetDefinition(
    name="my_dataset",
    has_files={"gaze": True, "precomputed_events": False, "precomputed_reading_measures": False},
    experiment=experiment,
    filename_format=filename_format,
    filename_format_schema_overrides=filename_format_schema_overrides,
    time_column=time_column, time_unit=time_unit,
    pixel_columns=pixel_columns)

dataset = pm.Dataset(definition=dataset_definition, path=dataset_paths)
