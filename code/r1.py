import pandas as pd

# Function to round the values in a column
def extract_two_decimal(string_value):
    return round(float(string_value), 2)

# Load the CSV file into a DataFrame
df = pd.read_csv(r'Narcis-New-Data.csv')


df = df.dropna()
# Assuming the column you want to modify is named 'your_column'
# df['open'] = df['open'].apply(extract_two_decimal)
# df['high'] = df['high'].apply(extract_two_decimal)
# df['low'] = df['low'].apply(extract_two_decimal)
# df['close'] = df['close'].apply(extract_two_decimal)
# df['percent'] = df['percent'].apply(extract_two_decimal)
# df['percent'] = df['percent'].astype(float).map(lambda x: f"{x}%")
df['volume'] = df['volume'].apply(extract_two_decimal)
#
# # Specify the column to modify (replace 'column_name' with the actual column name)
df['volume'] = df['volume'].astype(str).str.extract(r'(\d{2})', expand=False) + 'M'
#
df = df[(df != 0).all(axis=1)]


# Save the modified DataFrame back to a CSV
df.to_csv('Narcis-New-Data.csv', index=False)

# Print the modified DataFrame to check
print(df.head())