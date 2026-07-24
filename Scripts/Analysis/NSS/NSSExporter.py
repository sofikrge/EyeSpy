"""Stage 1 -> Stage 2 handoff: cleaned per-session event CSVs into one fixation parquet.

Concatenates every s_<session>_<participant>.csv, keeps the fixations, and classifies
each one by experimental phase and condition into the image_type the NSS analysis works
in (mooney_post_intact / mooney_post_not_intact / disamb_intact / disamb_not_intact).
Fixations that fall in no valid class are dropped.

Two derived columns are added here rather than in NSS.py, because both need the phase
boundaries only Stage 1 carries:
  - Mooney_Half   Early/Late half of the 3 s Mooney window, from onset - mooney_start.
  - trial_number  renumbered unique within a session: Experiment keeps 1-133, Extra gets
                  +300 (301-452), so trials also sort in true session order.

Numeric columns are cast to Float64 first, or sessions disagree on Int64 vs Float64 and
the schemas will not merge.

Reads:  data/events_cleaned/s_<session>_<participant>.csv   (CompleteRun.py)
Writes: data/NSS_all_fixations_clean.parquet
        columns: ImageName, session, image_type, participant, x_deg_centered, y_deg,
                 condition, trial_number, awareness, Mooney_Half, block_type
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

print(f"\nExporting fixations from {INPUT_DIR}...")

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
full_df = pl.concat(dfs, how="diagonal_relaxed").with_columns([
    pl.col("onset").cast(pl.Float64, strict=False),
    pl.col("mooney_start").cast(pl.Float64, strict=False),  # MSG-marker Mooney onset; used for the temporal half split
    # Renumber Extra-block trials (+300): trial_number restarts at 1 in the Extra
    # block, so without an offset an Experiment and an Extra viewing of the same
    # image can share a (participant, trial_number) key and be merged into one
    # scoring unit downstream. The offset makes trial_number unique within a
    # session AND sort in true session order (the Extra block ran after the
    # Experiment block), which the Experiment_Half median split relies on.
    pl.when(pl.col("block_type") == "Extra")
      .then(pl.col("trial_number").cast(pl.Int64) + 300)
      .otherwise(pl.col("trial_number").cast(pl.Int64))
      .cast(pl.Utf8)
      .alias("trial_number"),
])

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
    .with_columns([
        # --- MOONEY TEMPORAL HALF ---
        # Split the 3 s Mooney window into two 1.5 s halves by fixation onset,
        # measured from the true Mooney stimulus onset (mooney_start). A fixation
        # is assigned by its onset: < 1500 ms -> "Early", else "Late".
        # Only Mooney fixations get a half; disambiguation rows stay null.
        pl.when(pl.col("image_type").str.starts_with("mooney"))
          .then(
              # Guard a null mooney_start explicitly: without this it would fall through
              # to "Late" (null < 1500 is null -> otherwise) and hide a missing onset.
              pl.when(pl.col("mooney_start").is_null())
                .then(pl.lit("Unknown"))
                .when((pl.col("onset") - pl.col("mooney_start")) < 1500)
                .then(pl.lit("Early"))
                .otherwise(pl.lit("Late"))
          )
          .otherwise(pl.lit(None))
          .alias("Mooney_Half")
    ])
    .select([
        "ImageName", "session", "image_type", "participant",
        "x_deg_centered", "y_deg", "condition", "trial_number",
        "awareness", "Mooney_Half", "block_type"
    ])
    .drop_nulls(subset=["image_type"])
)

# 7. Save
export_df.write_parquet(OUTPUT_FILE)

n_unknown = export_df.filter(pl.col("Mooney_Half") == "Unknown").height
if n_unknown:
    print(f"WARNING: {n_unknown} Mooney fixations have a null mooney_start (Mooney_Half='Unknown').")

print(f"Exported {len(export_df)} rows to {OUTPUT_FILE}")
print(export_df.head(5))