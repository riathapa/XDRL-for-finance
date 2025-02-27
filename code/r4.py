import pandas as pd
# Load your CSV
df = pd.read_csv("Narcis-New-Data.csv")

# Flip the DataFrame
df_flipped = df.iloc[::-1].reset_index(drop=True)

# Save it back to CSV (optional)
df_flipped.to_csv("Narcis-New-Data.csv", index=False)

# Display the flipped DataFrame
print(df_flipped.head())