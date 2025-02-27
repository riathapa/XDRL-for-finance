import requests
import pandas as pd
from datetime import datetime

# Your NewsAPI key
api_key = 'fe11778863374c1b9cf29279dce59f28'  # Replace with your NewsAPI key


# Function to fetch news for a given stock symbol
def get_stock_news(symbol, start_date, end_date):
    # NewsAPI endpoint for fetching news articles
    url = f'https://newsapi.org/v2/everything'

    # Define the query parameters
    params = {
        'q': symbol,  # Stock symbol as a keyword (e.g., AAPL for Apple)
        'from': start_date,  # Start date in 'YYYY-MM-DD' format
        'to': end_date,  # End date in 'YYYY-MM-DD' format
        'apiKey': api_key,  # API Key
        'language': 'en',  # Optional: specify language (English)
        'sortBy': 'publishedAt',  # Sort articles by publication date
        'pageSize': 100,  # Max number of articles to fetch
    }

    # Send GET request to NewsAPI
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        articles = response.json().get('articles', [])

        # If articles are found, format them into a DataFrame
        if articles:
            news_df = pd.DataFrame(articles)
            return news_df
        else:
            print('No articles found for this stock in the given date range.')
            return None
    else:
        print(f"Error fetching news: {response.status_code}")
        return None


# Define stock symbol and date range
symbol = 'XOM'  # Example stock symbol (can be any stock ticker like AAPL, TSLA, etc.)
start_date = '2020-01-01'  # Start date in 'YYYY-MM-DD' format
end_date = '2021-01-31'  # End date in 'YYYY-MM-DD' format

# Fetch stock news
stock_news = get_stock_news(symbol, start_date, end_date)

# Show the news
if stock_news is not None:
    print(stock_news[['publishedAt', 'title', 'source', 'description']])

    # Optional: Save the news to a CSV file
    stock_news.to_csv(f'{symbol}_news_{start_date}_to_{end_date}.csv', index=False)
else:
    print('No news to display.')