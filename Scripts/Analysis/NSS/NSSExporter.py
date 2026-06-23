# OldNSSExporter.py

"""
Processes cleaned eye-tracking fixation data from multiple participant CSV files
and exports consolidated fixation data with image classification.
This script:
1. Reads cleaned event CSV files from data/events_cleaned/ (format: s_[session]_[participant].csv)
2. Extracts participant and session metadata from filenames using regex
3. Standardizes numeric columns to Float64 to prevent schema conflicts during concatenation
4. Filters for fixation events and classifies images based on experimental phase and condition
5. Extracts and normalizes gaze coordinates (x_deg_centered, y_deg)
6. Exports processed fixations to a Parquet file for further analysis in NSSAnalyses script
Output:
    Parquet file with columns: ImageName, session, image_type, participant, x_deg_centered, y_deg
    Only includes fixations with valid image_type classifications (mooney, disamb_intact, disamb_not_intact)
Note:
    Requires prior execution of EyeDrops.py to generate input CSV files.
    Handles mixed data types (Int64 vs Float64) in risky columns by casting to Float64.
"""

import polars as pl
import os
import re

# === CONFIG ===
INPUT_DIR = "data/events_cleaned"
OUTPUT_FILE = "data/NSS_all_fixations_clean.parquet"

# Regex to extract metadata from filename "s_C_101.csv"
# Group 1 = Session (Letters), Group 2 = Participant (Digits)
FILENAME_PATTERN = re.compile(r"s_([A-Za-z]+)_(\d+)\.csv")

print(f"\nStarting NSS Export from {INPUT_DIR}...")

dfs = []
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])

if not files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}. Did you run EyeDrops.py?")

for filename in files:
    # 1. Parse Metadata from Filename
    match = FILENAME_PATTERN.match(filename)
    if not match:
        print(f"Skipping non-conforming file: {filename}")
        continue
    
    session_id, part_id = match.groups()
    filepath = os.path.join(INPUT_DIR, filename)
    
    # 2. Read CSV (Infer schema from all rows to catch mixed NaNs/Ints)
    try:
        df = pl.read_csv(filepath, infer_schema_length=10000)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # 3. Add Metadata Columns
    df = df.with_columns([
        pl.lit(str(part_id)).alias("participant"),
        pl.lit(str(session_id)).alias("session")
    ])

    # 4. FIX: Force Risky Columns to Float64
    # This prevents the "Int64 vs Float64" crash when merging participants 
    # who have data (Ints) vs those who have NaNs (Floats).
    risky_cols = ["response_PAS_Q", "DidRespondPas", "NumRepetitionFixationFail", "BlockNum"]
    cast_exprs = [pl.col(c).cast(pl.Float64) for c in risky_cols if c in df.columns]
    
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    dfs.append(df)

# 5. Combine All Data
print(f"Merging {len(dfs)} participant files...")
full_df = pl.concat(dfs, how="diagonal_relaxed").with_columns(pl.col("onset").cast(pl.Float64, strict=False))

# 6. Transform & Select
# Note: 'location' in the CSV is a string like "[12.34, 56.78]".
# We must parse it back to floats.

print("Processing coordinates and assigning image types...")

export_df = (
    full_df
    .filter(pl.col("name") == "fixation")
    .with_columns([
        # --- USE CLEAN COORDINATES ---
        # We use the 'x' and 'y' columns we saved in EyeDrops.py
        pl.col("x").cast(pl.Float64).alias("x_deg_centered"),
        pl.col("y").cast(pl.Float64).alias("y_deg"),

        # --- IMAGE TYPE LOGIC (FIXED: Exact matching) ---
        pl.when((pl.col("phase") == "mooney") & 
                (pl.col("condition").str.to_lowercase() == "intact"))
          .then(pl.lit("mooney_post_intact"))

        .when((pl.col("phase") == "mooney") & 
                (pl.col("condition").str.to_lowercase().is_in(["not_intact", "scrambled"])))
          .then(pl.lit("mooney_post_scrambled"))
          
          .when((pl.col("phase") == "disambiguation") & 
                (pl.col("condition").str.to_lowercase().is_in(["not_intact", "scrambled"])))
          .then(pl.lit("disamb_not_intact"))
          
          .when((pl.col("phase") == "disambiguation") & 
                (pl.col("condition").str.to_lowercase() == "intact"))
          .then(pl.lit("disamb_intact"))
          
          .otherwise(pl.lit(None))
          .alias("image_type")
    ])
    .select([
        "ImageName", "session", "image_type", "participant",
        "x_deg_centered", "y_deg", "condition", "trial_number",
        "awareness", "dominant_eye"
    ])
    .drop_nulls(subset=["image_type"])
)

# 7. Save
export_df.write_parquet(OUTPUT_FILE)

print(f"Success! Exported {len(export_df)} rows to: {OUTPUT_FILE}")
print(export_df.head(5))