import polars as pl
from polars.testing import assert_frame_equal

#%% Check if changes made to preprocessing steps have altered the final outputs by comparing key CSV files before and after edits. This serves as a regression test to ensure that the preprocessing pipeline still produces the same results after modifications.
def run_regression_checks():
    # 1. Check all_events_cleaned.csv
    print("Checking all_events_cleaned.csv...")
    df_old_cleaned = pl.read_csv("data/OLD events_cleaned/all_events_cleaned.csv")
    df_new_cleaned = pl.read_csv("data/events_cleaned/all_events_cleaned.csv")
    
    assert_frame_equal(df_old_cleaned, df_new_cleaned)
    print("✅ Cleaned events DataFrames are 100% identical!\n")

    # 2. Check blink_spatial_filtering.csv
    print("Checking blink_spatial_filtering.csv...")
    df_old_qc = pl.read_csv("OLD DataQualityChecks/blink_spatial_filtering.csv")
    df_new_qc = pl.read_csv("DataQualityChecks/blink_spatial_filtering.csv")
    
    assert_frame_equal(df_old_qc, df_new_qc)
    print("✅ Blink spatial filtering QC DataFrames are 100% identical!\n")

if __name__ == "__main__":
    run_regression_checks()