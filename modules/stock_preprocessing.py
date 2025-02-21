import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from textblob import TextBlob  # pip install textblob
import logging
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

logger = logging.getLogger(__name__)

# --- Market Sentiment Functions ---
def scrape_headlines(url, headers=None):
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Adjust the tag/selector as needed for each website.
        headlines = [h.get_text(strip=True) for h in soup.find_all("h3")]
        return headlines
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return []

def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        analysis = TextBlob(headline)
        sentiments.append(analysis.sentiment.polarity)
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        if avg_sentiment > 0.1:
            return "Bullish"
        elif avg_sentiment < -0.1:
            return "Bearish"
        else:
            return "Neutral"
    return "Neutral"

def get_market_sentiment():
    urls = [
        "https://finance.yahoo.com",
        "https://www.cnbc.com",
        "https://www.bloomberg.com",
        "https://www.marketwatch.com",
    ]
    all_headlines = []
    for url in urls:
        headlines = scrape_headlines(url)
        all_headlines.extend(headlines)
    sentiment = analyze_sentiment(all_headlines)
    return sentiment, all_headlines

# --- Stock Data Collection Functions ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m", prepost=True)
        data["Ticker"] = ticker
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_sp500_stocks_data(ticker_list):
    df_list = []
    for ticker in ticker_list:
        df = fetch_stock_data(ticker)
        if not df.empty:
            df_list.append(df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info("Combined stock data fetched successfully.")
        return combined_df
    else:
        logger.error("No stock data fetched.")
        return pd.DataFrame()

# --- Filtering and Technical Indicator Functions ---
def filter_stocks(data, min_volume=1e6):
    filtered = data[data["Volume"] >= min_volume]
    return filtered

def calculate_technical_indicators(data):
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def generate_clean_stock_dataset(ticker_list):
    raw_data = fetch_sp500_stocks_data(ticker_list)
    if raw_data.empty:
        logger.error("No stock data available.")
        return raw_data
    raw_data = calculate_technical_indicators(raw_data)
    clean_data = filter_stocks(raw_data, min_volume=1e6)
    return clean_data

class StockPreprocessor:
    """
    Handles preprocessing of stock data including filtering, cleaning,
    and basic metric calculations
    """
    
    def __init__(self):
        self.min_volume = 1_000_000  # 1M shares minimum volume
        self.max_pe = 50
        self.required_columns = [
            'symbol', 'price', 'volume', 'pe_ratio', 
            'eps_growth', 'market_cap', 'sector'
        ]

    def filter_low_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out stocks with volume below threshold"""
        return df[df['volume'] >= self.min_volume].copy()

    def filter_high_pe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out stocks with P/E ratio above threshold"""
        return df[
            (df['pe_ratio'] <= self.max_pe) & 
            (df['pe_ratio'] > 0)
        ].copy()

    def filter_negative_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out stocks with negative EPS growth"""
        return df[df['eps_growth'] > 0].copy()

    def calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic financial metrics"""
        try:
            df['market_to_book'] = df['market_cap'] / df['book_value']
            df['debt_to_equity'] = df['total_debt'] / df['total_equity']
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']
            df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe has required columns"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main processing pipeline"""
        try:
            if not self.validate_data(df):
                raise ValueError("Data validation failed")

            # Create a copy to avoid modifying original
            processed_df = df.copy()

            # Apply filters
            processed_df = self.filter_low_volume(processed_df)
            processed_df = self.filter_high_pe(processed_df)
            processed_df = self.filter_negative_growth(processed_df)

            # Calculate additional metrics
            processed_df = self.calculate_basic_metrics(processed_df)

            # Remove duplicates and null values
            processed_df = processed_df.drop_duplicates()
            processed_df = processed_df.dropna(subset=self.required_columns)

            logger.info(f"Processed {len(processed_df)} stocks successfully")
            return processed_df

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise
