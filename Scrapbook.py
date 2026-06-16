import polars as pl

# Re-run a stripped down version of the diagonal_relaxed merge
import os, re
INPUT_DIR = "data/events_cleaned"
FILENAME_PATTERN = re.compile(r"s_([A-Za-z]+)_(\d+)\.csv")
dfs = []
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
for filename in files:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        continue
    df = pl.read_csv(os.path.join(INPUT_DIR, filename), infer_schema_length=10000)
    dfs.append(df)

full_df = pl.concat(dfs, how="diagonal_relaxed")
fixations_only = full_df.filter(pl.col("name") == "fixation")

print(f"Total fixation rows before image_type/null filtering: {len(fixations_only)}")
print(f"Nulls in 'phase': {fixations_only['phase'].is_null().sum()}")
print(f"Nulls in 'condition': {fixations_only['condition'].is_null().sum()}")