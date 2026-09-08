"""Single source of truth for the whole pipeline. This is the only file to edit.

Both stages read their parameters from here and configure nothing themselves:

    Stage 1 (preprocessing)   screen geometry, thresholds, blink handling, marker
                              codes, exclusions, and the pymovements dataset
    Stage 2 (NSS analysis)    canvas size, pixels-per-degree, subject thresholds

Anything that varies per run rather than per project stays out of this file: the
Mooney split, trial set and blink mode are chosen at runtime (see NSSPaths.py), and
the output paths follow from them.

One thing to know before changing anything: DEBUG gates I/O, not just logging. With
DEBUG on, the intermediate data/events/ CSVs and every QC figure under
DataQualityChecks/ get written.
"""

#%% Imports
import os
import re
import polars as pl
import matplotlib.pyplot as plt
import pymovements as pm
from pathlib import Path

#%% ============================ DATASET SELECTION ==============================
# Which dataset this run works on. CompleteRun.py sets EYESPY_DATASET from its
# DATASET toggle; set it by hand (EYESPY_DATASET=rep) when running a Stage-2 script
# on its own. "rep" swaps in the data_rep/ folder and applies Settings_rep.py at the
# bottom of this file, so every parameter below can be overridden per dataset.
REP       = os.environ.get("EYESPY_DATASET", "") == "rep"
DATA_ROOT = "data_rep" if REP else "data"
_SUFFIX   = "_rep" if REP else ""

#%% =========================== STAGE 1: PREPROCESSING ===========================

DEBUG = True

dataset_paths = pm.DatasetPaths(
    root=f'{DATA_ROOT}/', 
    raw='raw', 
    preprocessed='preprocessed', 
    events='events')

SCREEN = { # adjust to your parameters
    "width_px": 1920, "height_px": 1080,
    "width_cm": 53.2, "height_cm": 29.8,
    "distance_cm": 74.0,
    "origin": "upper left",
    "sampling_rate": 1000}

experiment = pm.gaze.Experiment(
    screen_width_px=SCREEN["width_px"], screen_height_px=SCREEN["height_px"],
    screen_width_cm=SCREEN["width_cm"], screen_height_cm=SCREEN["height_cm"],
    distance_cm=SCREEN["distance_cm"], origin=SCREEN["origin"],
    sampling_rate=SCREEN["sampling_rate"])

filename_format = {'gaze': r's_{session_id:s}_{participant_id:d}.asc'}
filename_format_schema_overrides = {'gaze': {'session_id': str, 'participant_id': str}}

time_column, time_unit = 'time', 'ms'
pixel_columns = ['x_right', 'y_right', 'x_left', 'y_left']

dataset_definition = pm.DatasetDefinition(
    name="my_dataset",
    has_files={"gaze": True, "precomputed_events": False, "precomputed_reading_measures": False},
    experiment=experiment,
    filename_format=filename_format,
    filename_format_schema_overrides=filename_format_schema_overrides,
    time_column=time_column, time_unit=time_unit,
    pixel_columns=pixel_columns)

dataset = pm.Dataset(definition=dataset_definition, path=dataset_paths)

# Folders (each writer creates it on demand, so nothing is made just by importing)
data_quality_folder = f'DataQualityChecks{_SUFFIX}/'

# Validation thresholds
VALIDATION_ACCURACY_AVG_THRESHOLD = 1.0  # degrees
VALIDATION_ACCURACY_MAX_THRESHOLD = 1.5  # degrees

# Offset value of eyes in visual degrees 
EYE_OFFSET = {"left": +5.44, "right": -5.44} 

RAW_DATA_DIR = os.path.join(DATA_ROOT, 'my_dataset', 'raw') # path to raw data for manual blink parsing
EVENTS_OUT_DIR = os.path.join(DATA_ROOT, 'events')
BEHAVIOURAL_DIR = Path(f"{DATA_ROOT}/my_dataset/behavioural") # path to behavioural data
EVENTS_CLEANED_DIR = os.path.join(DATA_ROOT, 'events_cleaned')

FIX_VELOCITY_THRESHOLD = 30.0  # degrees per second
MIN_FIX_DURATION_MS = 50      # minimum fixation length

BUFFER_FIX = 51   # 51ms for fixations
BUFFER_SAC = 60   # 50ms + 10ms for saccades

# --- Blink handling: two mutually exclusive modes
#   False -> drop events overlapping a blink (the primary, preregistered method)
#   True  -> PCHIP-interpolate position across short blinks before event detection,
#            so a blink-spanning fixation survives as one fixation
# Rationale and sources for all three values: REFERENCES.md.
INTERPOLATE_BLINKS = True
MAX_BLINK_INTERP_MS = 150   # longer gaps are track loss, left as gaps
BLINK_MARGIN_MS = 200       # each blink is widened by this before interpolating

IMAGE_SIZE_DEG = (9.99, 7.50)
CENTER_RADIUS_DG = 1.5 #shaked's value
HX, HY = IMAGE_SIZE_DEG[0] / 2, IMAGE_SIZE_DEG[1] / 2

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
    'NumRepetitionFixationFail': 'NumRepetitionFixationFail',
    'response_PAS_Q': 'response_PAS_Q',
}

EXCLUDE_SUBJECTS = [ 
# 104, 106, 109, 110, 112, 118, 120, # left dominant eye
]

# Format: { ParticipantID: ['SessionLetter'] }
EXCLUDE_SESSIONS = {
#    104: ['U'], # unfocused eyes sometimes
#    105: ['U'], # unfocused eyes sometimes
#    106: ['U'], # unfocused eyes sometimes
    107: ['U'], # low PAS 0 trials
    111: ['U'], # low PAS 0 trials
    110: ['U'], # low PAS 0 trials
    118: ['U'], # low PAS 0 trials
#    112: ['C'], # unfocused eyes sometimes
}

# Exclude specific BLOCKS per session per participant
# Format: { ParticipantID: { 'SessionID': [BlockNums] } }
EXCLUDE_BLOCKS = {
#    112: {'U': [4, 5]},             # Unfocused eyes sometimes
    117: {'C': [1]},                # Technical mistake
    119: {'U': [1, 2, 3, 4, 5, 6]}  # Unfocused eyes sometimes
}

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
MASK_PPD     = 48.55            # pixels per visual degree
SIGMA        = MASK_PPD / 2.0   # Gaussian blur radius for the fixation maps

FIX_FILE = Path(f"{DATA_ROOT}/NSS_all_fixations_clean.parquet")   # written by NSSExporter.py
ANALYSES_ROOT = f"analysesresults{_SUFFIX}"   # NSSPaths.py builds its NSS_* folders in here

# How much data an image needs before it is scored at all.
MIN_SUBJ_PER_IMAGE_NSS    = 2    # within-phase: minimum subjects per image
MIN_SUBJ_PER_IMAGE_CROSS  = 2    # cross-phase: minimum Mooney subjects per image
MIN_IMAGES_PER_CELL_CROSS = 15   # cross-phase: minimum valid scores per participant cell

# Missing reference maps: "permissive" averages whatever is present,
# "matlab_strict" requires both references or scores the image NaN.
NAN_POLICY_CROSS = "permissive"

DISPERSION_DDOF = 0   # population sd, for MATLAB parity (REFERENCES.md)

# Reporting thresholds, used only by Scripts/Analysis/Checks/.
MIN_IMAGES_PER_PARTICIPANT = 15   # ImagePerParticipant.py flags anyone below this
MIN_FIX_PER_PARTICIPANT    = 20   # LeftBiasPerParticipant.py ignores thinner cells


#%% ======================= REPLICATION-DATASET OVERRIDES ========================
# Applied last, so Settings_rep.py can override anything above for the data_rep/
# dataset. The pilot run never touches it.
if REP:
    print("[Settings] EYESPY_DATASET=rep -> applying Settings_rep.py overrides")
    from Settings_rep import *  # noqa: F401,F403
