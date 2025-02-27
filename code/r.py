import yfinance as yf
import time
# yf.pdr_override()
import pandas as pd


def download_stock_data(tickers, start_date, end_date, filename="stock_data.csv"):
    """
    Downloads historical stock data for the given tickers and saves it as a CSV file.

    Parameters:
        tickers (list): A list of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        filename (str): Name of the output CSV file.
    """
    data_list = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"⚠️ No data found for {ticker}")
            continue

        df["Ticker"] = ticker
        df["Percent Change"] = df["Close"].pct_change() * 100  # Calculate percent change
        df = df[["Ticker", "Open", "High", "Low", "Close", "Volume", "Percent Change"]]
        data_list.append(df)
        time.sleep(60)

    if data_list:
        final_df = pd.concat(data_list)
        final_df.to_csv("Siylis-New-Data.csv", index=True)
        print(f"✅ Data saved successfully in {filename}")
    else:
        print("❌ No data available for the given stocks.")


# Example Usage
tickers = ["TSM", "MRVL", "NVDA", "RTS", "GM"]
start_date = "2020-01-01"
end_date = "2022-02-01"

download_stock_data(tickers, start_date, end_date)