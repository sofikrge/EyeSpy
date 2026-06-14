# Scripts/Analysis/NSSExporter.py

"""
Processes cleaned eye-tracking fixation data from multiple participant CSV files
and exports consolidated fixation data with image classification.

FLAG: BLEND_TRIALS
  - True  (original behaviour): fixations from all presentations of the same image
           by the same participant are pooled. The unit of analysis is
           participant × image × session.
  - False (per-trial mode):     each presentation is kept separate.
           The unit of analysis is participant × image × trial × session.
           'trial_number' is included in the parquet so NSS.py can group by it.

Output parquet columns (both modes):
    ImageName, session, image_type, participant, x_deg_centered, y_deg,
    condition, trial_number
    (trial_number is the raw trial index from the cleaned event CSV)
"""

import polars as pl
import os
import re

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Set to True  → blend all fixations for participant × image (original behaviour)
# Set to False → keep each trial separate (recommended when images repeat)
BLEND_TRIALS = False

FILENAME_PATTERN = re.compile(r"s_([A-Za-z]+)_(\d+)\.csv")

# Columns that can be Int64 in some files and Float64 in others → force Float64
RISKY_COLS = ["response_PAS_Q", "DidRespondPas", "NumRepetitionFixationFail", "BlockNum"]

# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def export_nss_fixations(
    input_dir: str = "data/events_cleaned",
    output_file: str = "data/NSS_all_fixations_clean.parquet",
    blend_trials: bool = BLEND_TRIALS,
) -> str:
    """
    Read all cleaned event CSVs, classify fixations by image type, and write
    a parquet file ready for NSSAnalysis.py.

    Parameters
    ----------
    input_dir   : folder containing s_<session>_<participant>.csv files
    output_file : destination parquet path
    blend_trials: see module docstring

    Returns
    -------
    output_file path (str)
    """

    mode_label = "BLEND mode (participant × image)" if blend_trials \
                 else "PER-TRIAL mode (participant × image × trial)"
    print(f"\nStarting NSS Export from '{input_dir}' [{mode_label}]...")

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{input_dir}'.")

    dfs = []
    for filename in files:
        match = FILENAME_PATTERN.match(filename)
        if not match:
            print(f"  Skipping non-conforming file: {filename}")
            continue

        session_id, part_id = match.groups()
        filepath = os.path.join(input_dir, filename)

        try:
            df = pl.read_csv(filepath, infer_schema_length=10000)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        # Tag with participant and session derived from filename
        df = df.with_columns([
            pl.lit(str(part_id)).alias("participant"),
            pl.lit(str(session_id)).alias("session"),
        ])

        # Homogenise mixed-type columns so concat doesn't crash
        cast_exprs = [pl.col(c).cast(pl.Float64) for c in RISKY_COLS if c in df.columns]
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No files could be read. Check the input directory.")

    print(f"  Merging {len(dfs)} participant files...")
    full_df = pl.concat(dfs, how="diagonal")   # diagonal = handle missing cols safely

    print("  Classifying image types and extracting coordinates...")

    export_df = (
        full_df
        .filter(pl.col("name") == "fixation")
        .with_columns([
            pl.col("x").cast(pl.Float64).alias("x_deg_centered"),
            pl.col("y").cast(pl.Float64).alias("y_deg"),

            # ── Image type classification ──────────────────────────────────
            pl.when(
                (pl.col("phase") == "mooney") &
                (pl.col("condition").str.to_lowercase() == "intact")
            ).then(pl.lit("mooney_post_intact"))

            .when(
                (pl.col("phase") == "mooney") &
                (pl.col("condition").str.to_lowercase().is_in(["not_intact", "scrambled"]))
            ).then(pl.lit("mooney_post_scrambled"))

            .when(
                (pl.col("phase") == "disambiguation") &
                (pl.col("condition").str.to_lowercase().is_in(["not_intact", "scrambled"]))
            ).then(pl.lit("disamb_not_intact"))

            .when(
                (pl.col("phase") == "disambiguation") &
                (pl.col("condition").str.to_lowercase() == "intact")
            ).then(pl.lit("disamb_intact"))

            .otherwise(pl.lit(None))
            .alias("image_type"),
        ])
        # Always keep trial_number — needed in per-trial mode and harmless in blend mode
        .select([
            "ImageName",
            "session",
            "image_type",
            "participant",
            "x_deg_centered",
            "y_deg",
            "condition",
            "trial_number",
        ])
        .drop_nulls(subset=["image_type"])
    )

    # ── In blend mode, drop trial_number so downstream code sees no trial dimension
    if blend_trials:
        export_df = export_df.drop("trial_number")
        print("  BLEND mode: trial_number dropped — fixations pooled per participant × image.")
    else:
        n_trials = export_df.select("trial_number").n_unique()
        print(f"  PER-TRIAL mode: {n_trials} unique trial numbers retained.")

    export_df.write_parquet(output_file)
    print(f"  ✅ Exported {len(export_df):,} fixation rows → {output_file}")
    return output_file