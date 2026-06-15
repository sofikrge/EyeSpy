import pandas as pd
from pathlib import Path

fixations = pd.read_parquet("data/NSS_all_fixations_clean.parquet")

mooney = fixations[
    (fixations["image_type"] == "mooney_post_intact") &
    (fixations["awareness"] == "conscious_unaware")
]

counts = (
    mooney.groupby("ImageName")["participant"]
    .nunique()
    .reset_index(name="n_participants")
    .sort_values("n_participants", ascending=False)
)

print(f"Total images with any conscious_unaware fixations: {len(counts)}")
print(f"Images with >= 2 participants: {(counts['n_participants'] >= 2).sum()}")
print(f"Images with exactly 1 participant: {(counts['n_participants'] == 1).sum()}")
print(f"\nDistribution of participant counts:")
print(counts["n_participants"].value_counts().sort_index())
print(f"\nTop 10 images by participant count:")
print(counts.head(10))