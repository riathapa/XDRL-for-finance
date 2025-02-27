import pandas as pd
import ast
import numpy as np

# Load CSV
df = pd.read_csv("ria_state_action_recopilation_cleaned.csv")

# Column that contains the nested arrays (replace 'your_column' with actual name)
column_name = "State"


# Function to process each row in the column
def expand_ones(array_str):
    try:
        # Convert string to Python list
        parsed_data = ast.literal_eval(array_str)

        # Loop through elements and replace `1.0` with `[1.0, 1.0, 1.0, 1.0]`
        for i in range(len(parsed_data[0])):
            if parsed_data[0][i] == [[1.0]]:  # Match the exact structure of `[[1.0]]`
                parsed_data[0][i] = [[1.0, 1.0, 1.0, 1.0]]  # Replace it

        return str(parsed_data)  # Convert back to string for storage in CSV

    except Exception as e:
        print(f"Error processing row: {array_str}, Error: {e}")
        return None  # Handle errors gracefully


# Apply transformation to the whole column
df[column_name] = df[column_name].apply(expand_ones)

# Save the cleaned DataFrame back to CSV
df.to_csv("ria_state_action_recopilation_cleaned2.csv", index=False)

# Display the cleaned DataFrame
print(df.head())