from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time

# Replace with your Alpha Vantage API key
api_key = 'LCCKL7LBXDAOFGGU'

# Initialize the TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# List of stock symbols
stock_symbols = ["XOM", "SHEL", "CVX", "OKE", "VLO"]
all_stock_data = []

# Define start and end dates
start_date = '2020-01-01'
end_date = '2021-12-31'

# Fetch and filter data for each stock
for symbol in stock_symbols:
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        # Filter by date range
        filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)].copy()
        filtered_data['Symbol'] = symbol  # Add a column to identify the stock
        filtered_data.reset_index(inplace=True)  # Reset index to make the date a column
        all_stock_data.append(filtered_data)
        print(f"Fetched and filtered data for {symbol}")
        time.sleep(12)  # Avoid rate limits
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Combine all the data into a single DataFrame
combined_data = pd.concat(all_stock_data, ignore_index=True)

# Save to CSV
combined_data.to_csv('Omar-New-Data.csv', index=False)

print("All data saved to 'all_stocks_data.csv'")