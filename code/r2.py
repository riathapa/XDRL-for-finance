import pandas as pd
combined_data = pd.read_csv(r'Siylis-New-Data.csv')
# combined_data = combined_data.rename(columns={
#     'date': 'date',
#     '4. close': 'close',
#     '1. open': 'open',
#     '2. high': 'high',
#     '3. low': 'low',
#     '5. volume': 'volume',
#     'Symbol': 'code'
# })
# combined_data = combined_data.sort_values(by='date', ascending=True).reset_index(drop=True)

combined_data['Date'] = pd.to_datetime(combined_data['date'])
combined_data.to_csv('Siylis-New-Data.csv', index=False)