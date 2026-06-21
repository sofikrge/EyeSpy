import pandas as pd
fix = pd.read_parquet("data/NSS_all_fixations_clean.parquet")
p8 = fix[(fix["participant"].astype(str) == "8") &
         (fix["image_type"] == "mooney_post_intact") &
         (fix["awareness"] == "unconscious_unaware")]
print(p8.groupby("ImageName").size())


