#%% Imports
import os
import polars as pl
import matplotlib.pyplot as plt
import pymovements as pm

#%% PyMovements Settings

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