# Settings.py

#%% Imports
import os
import re
import polars as pl
import matplotlib.pyplot as plt
import pymovements as pm
from pathlib import Path

#%% Settings

DEBUG = True

dataset_paths = pm.DatasetPaths(
    root='data/', 
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

# Folders
data_quality_folder = 'DataQualityChecks/' ; os.makedirs(data_quality_folder, exist_ok=True)

# Validation thresholds
VALIDATION_ACCURACY_AVG_THRESHOLD = 1.0  # degrees
VALIDATION_ACCURACY_MAX_THRESHOLD = 1.5  # degrees

# Offset value of eyes in visual degrees 
EYE_OFFSET = {"left": +5.44, "right": -5.44} 

RAW_DATA_DIR = os.path.join('data', 'my_dataset', 'raw') # path to raw data for manual blink parsing
EVENTS_OUT_DIR = os.path.join('data', 'events')
BEHAVIOURAL_DIR = Path("data/my_dataset/behavioural") # path to behavioural data
EVENTS_CLEANED_DIR = os.path.join('data', 'events_cleaned')

FIX_VELOCITY_THRESHOLD = 30.0  # degrees per second
MIN_FIX_DURATION_MS = 50      # minimum fixation length

BUFFER_FIX = 51   # 51ms for fixations
BUFFER_SAC = 60   # 50ms + 10ms for saccades

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

EXCLUDE_SUBJECTS = [] 
# Format: { ParticipantID: ['SessionLetter'] } 

EXCLUDE_SESSIONS = {
#    104: ['U'], # unfocused eyes sometimes
#    105: ['U'], # unfocused eyes sometimes
#    106: ['U'], # unfocused eyes sometimes
    107: ['U'], # low PAS 0 trials
    110: ['U'], # low PAS 0 trials
    111: ['U'], # low PAS 0 trials
#    112: ['C'], # unfocused eyes sometimes
    118: ['U'] # low PAS 0 trials
}

# Exclude specific BLOCKS per session per participant
# Format: { ParticipantID: { 'SessionID': [BlockNums] } }
EXCLUDE_BLOCKS = {
#    112: {'U': [4, 5]},             # Unfocused eyes sometimes
    117: {'C': [1]},                # Technical mistake
#    119: {'U': [1, 2, 3, 4, 5, 6]}  # Unfocused eyes sometimes
}

# PLotting colors
FILTER_PALETTE = ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d']
PHASE_PALETTE  = ['#b3cde3', '#8c96c6', '#88419d']