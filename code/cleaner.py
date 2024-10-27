import pandas as pd
import numpy as np
import ast
import json


def clean_state_action_csv(input_file, output_file, config_file):
    # Load the config.json file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract values of M, L, and N from the config
    M = len(config['session']['codes']) + 1  # Number of assets + 1 for cash or risk-free asset
    L = int(config['session']['agents'][2])  # Look-back window length
    N = len(config['session']['features'])  # Number of features per asset
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Function to parse the string representation of arrays
    def parse_array(array_str):
        return np.array(ast.literal_eval(array_str.replace('\n', '').replace('  ', ',').replace(' ', ',')))
    
    # print(df.head())
    
    # Parse the State and Action columns
    df['State'] = df['State'].apply(parse_array)
    df['Action'] = df['Action'].apply(parse_array)

    # print(df.head())
    
   # Flatten the State and Action columns
    flattened_data = []
    for _, row in df.iterrows():
        action = row['Action'].flatten()
        state = row['State'].flatten()
        flattened_row = np.concatenate((action, state))
        flattened_data.append(flattened_row)
    
    # print(flattened_data)
    
    # Create column names
    # Action columns with asset names
    action_columns = []
    for i in range(M):
        asset_name = config['session']['codes'][i-1] if i > 0 else 'Cash'
        column_name = f'Action_{asset_name}'
        action_columns.append(column_name)
    state_columns = []
    for i in range(M):
        for j in range(L):
            for k in range(N):
                # Add the feature name and asset name to the column
                feature_name = config['session']['features'][k]
                asset_name = config['session']['codes'][i-1] if i > 0 else 'Cash'
                column_name = f'State_{asset_name}_{feature_name}_L{j+1}'
                state_columns.append(column_name)
    columns = action_columns + state_columns
    # print(columns)
    # Create a new DataFrame with the flattened data
    cleaned_df = pd.DataFrame(flattened_data, columns=columns)
    
    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_file, index=False)


def merge_state_action_results(cleaned_state_action_file, results_file, output_file,config_file):
    # Load the config.json file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract values of M, L, and N from the config
    M = len(config['session']['codes']) + 1  # Number of assets + 1 for cash or risk-free asset
    
    # Load the cleaned state-action CSV
    state_action_df = pd.read_csv(cleaned_state_action_file)
    
    # Load the results CSV
    results_df = pd.read_csv(results_file)
    
    # Drop the 'weight' column from the results CSV if it exists
    if 'Weight' in results_df.columns:
        results_df = results_df.drop(columns=['Weight'])
    
    price_columns = []
    # Divide the price column in M columns by the comma
    for i in range(M):
        # Add the asset name to the column from the config
        asset_name = config['session']['codes'][i-1] if i > 0 else 'Cash'
        price_columns.append(f'Price_{asset_name}')
    results_df[price_columns] = results_df['Price'].str.split(',', expand=True)
    results_df.drop('Price', axis=1, inplace=True)

    # Drop the first column of the results_df
    results_df.drop(results_df.columns[0], axis=1, inplace=True)

    # Merge the two dataframes on their indices (assuming they align correctly)
    merged_df = pd.concat([state_action_df, results_df], axis=1)
    
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)



# Example usage
input_file = 'state_action_recopilation.csv'
output_file = 'cleaned_state_action.csv'
config_file = 'config.json'

clean_state_action_csv(input_file, output_file, config_file)

# Example usage
cleaned_state_action_file = 'cleaned_state_action.csv'
results_file = 'result1-57.064914695339375.csv'
output_file = 'state_action_results.csv'
config_file = 'config.json'

# merge_state_action_results(cleaned_state_action_file, results_file, output_file,config_file)
