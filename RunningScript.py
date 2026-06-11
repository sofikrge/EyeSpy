#!/usr/bin/env python3
# RunningScript.py
#%% Imports
import pymovements as pm
import Scripts.Preprocessing.Preprocessing as prep
import Settings as settings

#%% How to use
"""
How to use:
- Type this in the terminal: alias run='python3 RunningScript.py'
- Type "run" in the terminal 
"""

#%%
def main():

    #%% Preprocessing
    global dataset

    print("\nLoading raw data...")
    settings.dataset.load()

    print("\nConverting to visual degrees...")
    settings.dataset.pix2deg()

    print("\nApplying eye offset correction...")
    settings.dataset = prep.shift_gaze_offset(settings.dataset, settings.EYE_OFFSET)

    print("\nAdding velocity column...")
    settings.dataset.pos2vel(method='fivepoint')

    print("\nFiltering and reporting validations...")
    prep.filter_and_report_validations(
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

    print("\nFiltering events (blinks and spatial bounds)...")
    prep.filter_events_blink_spatial(
        dataset=settings.dataset, raw_data_dir=settings.RAW_DATA_DIR,
        buffer_fix=settings.BUFFER_FIX, buffer_sac=settings.BUFFER_SAC,
        hx=settings.HX, hy=settings.HY,
        center_radius_dg=settings.CENTER_RADIUS_DG,
        data_quality_folder=settings.data_quality_folder,
        debug=settings.DEBUG,
        image_size_deg=settings.IMAGE_SIZE_DEG,
        filter_palette=settings.FILTER_PALETTE)
    
    print("\nAssigning trial metadata and image phases to events...")
    prep.assign_trial_metadata_and_phases(
        dataset=settings.dataset, raw_data_dir=settings.RAW_DATA_DIR,
        behavioural_dir=settings.BEHAVIOURAL_DIR, events_out_dir=settings.EVENTS_OUT_DIR,
        trial_labels=settings.TRIAL_LABELS,
        asc_patterns=settings.ASC_PATTERNS,
        section_to_block=settings.SECTION_TO_BLOCK,
        debug=settings.DEBUG,
        data_quality_folder=settings.data_quality_folder,
        phase_palette=settings.PHASE_PALETTE)

    print("\nApplying final behavioral filtering...")
    combined_df = prep.apply_behavioral_filters_and_save(
        dataset=settings.dataset,
        output_dir=settings.EVENTS_CLEANED_DIR,
        exclude_subjects=settings.EXCLUDE_SUBJECTS,
        exclude_sessions=settings.EXCLUDE_SESSIONS,
        exclude_blocks=settings.EXCLUDE_BLOCKS)

    #%% NSS Analysis

if __name__ == "__main__":
    main()