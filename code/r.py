import yfinance as yf
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
        df = df[["close", "open", "high", "low", "volume", "percent", "Ticker"]]
        data_list.append(df)

    if data_list:
        final_df = pd.concat(data_list)
        final_df.to_csv("stock_data.csv", index=True)
        print(f"✅ Data saved successfully in {filename}")
    else:
        print("❌ No data available for the given stocks.")


# Example Usage
tickers = ["BMW", "FORD", "GM", "MBG.DE", "TSLA"]
start_date = "2024-01-01"
end_date = "2024-02-01"

download_stock_data(tickers, start_date, end_date)