# modules/data_collection.py
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()

def fetch_data_for_ticker(ticker):
    """
    Fetch pre-market data for a given ticker and tag the DataFrame with the ticker symbol.
    """
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="1d", interval="1m", prepost=True)
        data['Ticker'] = ticker  # add ticker identifier to the DataFrame
        logger.info(f"Data for {ticker} fetched successfully.")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_multiple_stocks(ticker_list):
    """
    Fetch data for a list of tickers and combine into a single DataFrame.
    """
    dfs = []
    for ticker in ticker_list:
        df = fetch_data_for_ticker(ticker)
        if df is not None and not df.empty:
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs)
        logger.info("Combined data for multiple tickers successfully.")
        return combined
    else:
        logger.error("No data fetched for any tickers.")
        return None

def fetch_cnbc_headlines(url="https://www.cnbc.com/markets/"):
    """
    Scrape CNBC headlines using a browser-like User-Agent.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h.get_text(strip=True) for h in soup.select('.Card-title')]
        logger.info(f"Fetched {len(headlines)} headlines from CNBC.")
        return headlines
    except Exception as e:
        logger.error(f"Error fetching CNBC headlines: {e}")
        raise

def save_data(data, filename):
    """
    Save a DataFrame to CSV.
    """
    try:
        data.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
