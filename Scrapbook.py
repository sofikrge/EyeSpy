import pandas as pd

# Load the file you are sending to Jamovi
df_jamovi = pd.read_csv("analysesresults/NSS/NSS_CrossPhase_LongFormat.csv")

# Filter for the problematic image
debug_1009 = df_jamovi[df_jamovi["Image"] == "1009.jpg"]

print(debug_1009[["Session", "Awareness", "NSS"]])

# Check if there are any NaNs in the NSS column for this image
print(f"Missing values for 1009: {debug_1009['NSS'].isna().sum()}")
