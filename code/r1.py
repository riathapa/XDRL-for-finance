import pandas as pd

# Function to round the values in a column
def extract_two_decimal(string_value):
    return round(float(string_value), 2)

# Load the CSV file into a DataFrame
df = pd.read_csv('../data/NarcisData.csv')

# Assuming the column you want to modify is named 'your_column'
df['close'] = df['close'].apply(extract_two_decimal)
df['open'] = df['open'].apply(extract_two_decimal)
df['high'] = df['high'].apply(extract_two_decimal)
df['low'] = df['low'].apply(extract_two_decimal)

# Save the modified DataFrame back to a CSV
df.to_csv('../data/NarcisData.csv', index=False)

# Print the modified DataFrame to check
print(df.head())