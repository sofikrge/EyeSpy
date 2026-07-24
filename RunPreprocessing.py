"""Stage 1: raw EyeLink recordings -> cleaned fixation/saccade event tables.

Top-to-bottom orchestration, written as `#%%` cells so it can also be stepped through
in an interactive window. Every parameter comes from Settings.py; nothing is configured
here. Run it from the project root:

    python3 RunPreprocessing.py

Reads:  data/my_dataset/raw/s_<SESSION>_<PID>.asc
        data/my_dataset/behavioural/expdata_<SESSION>_<PID>.mat
Writes: data/events_cleaned/s_<SESSION>_<PID>.csv  +  all_events_cleaned.csv
        DataQualityChecks/  (figures, only when DEBUG is on in Settings.py)
"""

#%% Imports
import Scripts.Preprocessing.BlinkInterpolation as blinks
import Scripts.Preprocessing.GazeCorrection as gaze
import Scripts.Preprocessing.EventFiltering as events
import Scripts.Preprocessing.TrialMetadata as trials
import Settings as settings

#%% Load data
global dataset

print("\nLoading raw data...")
settings.dataset.load()

#%% Optional blink interpolation (must run on raw pixels, before pix2deg)
if settings.INTERPOLATE_BLINKS:
    settings.dataset = blinks.interpolate_blink_gaps(
        settings.dataset, settings.MAX_BLINK_INTERP_MS,
        settings.SCREEN["sampling_rate"], settings.BLINK_MARGIN_MS)

#%% Preprocess
print("\nConverting to visual degrees...")
settings.dataset.pix2deg()

print("\nApplying eye offset correction...")
settings.dataset = gaze.shift_gaze_offset(settings.dataset, settings.EYE_OFFSET)

print("\nAdding velocity column...")
settings.dataset.pos2vel(method='fivepoint')

print("\nFiltering and reporting validations...")
events.filter_and_report_validations(
    settings.dataset,settings.data_quality_folder,
    settings.VALIDATION_ACCURACY_AVG_THRESHOLD,
    settings.VALIDATION_ACCURACY_MAX_THRESHOLD)

settings.dataset.save_preprocessed(preprocessed_dirname='preprocessed', extension='csv')

print("\nDetecting events (Fixations and Saccades)...")
settings.dataset.detect_events(timesteps='time', method='ivt', velocity_threshold=settings.FIX_VELOCITY_THRESHOLD, 
                                minimum_duration=settings.MIN_FIX_DURATION_MS)

settings.dataset.detect_events('microsaccades')

print("\nComputing extra event properties...")
settings.dataset.compute_event_properties(['location', 'amplitude', 'peak_velocity', 'dispersion', 'disposition'])

#%% Blink and spatial filtering
print("\nFiltering events (blinks and spatial bounds)...")
events.filter_events_blink_spatial(
    dataset=settings.dataset, raw_data_dir=settings.RAW_DATA_DIR,
    buffer_fix=settings.BUFFER_FIX, buffer_sac=settings.BUFFER_SAC,
    hx=settings.HX, hy=settings.HY,
    center_radius_dg=settings.CENTER_RADIUS_DG,
    data_quality_folder=settings.data_quality_folder,
    debug=settings.DEBUG,
    image_size_deg=settings.IMAGE_SIZE_DEG,
    filter_palette=settings.FILTER_PALETTE,
    skip_blink_filter=settings.INTERPOLATE_BLINKS)  # blinks already interpolated -> keep those events

#%% Assign trial metadata 
print("\nAssigning trial metadata and image phases to events...")
trials.assign_trial_metadata_and_phases(
    dataset=settings.dataset, raw_data_dir=settings.RAW_DATA_DIR,
    behavioural_dir=settings.BEHAVIOURAL_DIR, events_out_dir=settings.EVENTS_OUT_DIR,
    trial_labels=settings.TRIAL_LABELS,
    asc_patterns=settings.ASC_PATTERNS,
    section_to_block=settings.SECTION_TO_BLOCK,
    debug=settings.DEBUG,
    data_quality_folder=settings.data_quality_folder,
    phase_palette=settings.PHASE_PALETTE)

#%% Behavioural filtering
print("\nApplying final behavioral filtering...")
trials.apply_behavioral_filters_and_save(
    dataset=settings.dataset,
    output_dir=settings.EVENTS_CLEANED_DIR,
    exclude_subjects=settings.EXCLUDE_SUBJECTS,
    exclude_sessions=settings.EXCLUDE_SESSIONS,
    exclude_blocks=settings.EXCLUDE_BLOCKS)

# %%
