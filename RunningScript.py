#!/usr/bin/env python3
#%% Imports
import pymovements as pm
import Preprocessing as prep
import Settings as settings

#%% How to use
"""
How to use:
- Type this in the terminal: alias run='python3 RunningScript.py'
- Type "run" in the terminal 
"""

#%%
def main():

    global dataset

    print("\nLoading raw data...")
    settings.dataset.load()

    print("\nConverting to visual degrees...")
    settings.dataset.pix2deg()

    print("\nAdding velocity column...")
    settings.dataset.pos2vel(method='fivepoint')

    print("\nFiltering and reporting validations...")
    prep.filter_and_report_validations(
        settings.dataset, 
        settings.data_quality_folder,
        settings.VALIDATION_ACCURACY_AVG_THRESHOLD,
        settings.VALIDATION_ACCURACY_MAX_THRESHOLD
    )
    
    settings.dataset.save_preprocessed(preprocessed_dirname='preprocessed', extension='csv')

    print("\nDetecting events (Fixations and Saccades)...")
    settings.dataset.detect_events(timesteps='time', method='ivt', velocity_threshold=settings.FIX_VELOCITY_THRESHOLD, minimum_duration=settings.MIN_FIX_DURATION_MS)
    settings.dataset.detect_events('microsaccades')

    print("\nComputing extra event properties...")
    settings.dataset.compute_event_properties(['location', 'amplitude', 'peak_velocity', 'dispersion', 'disposition'])
    
if __name__ == "__main__":
    main()