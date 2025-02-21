import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger
from datetime import datetime, timedelta
import pandas_datareader.data as web

class MarketDataCollector:
    """Collects market data from various sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = timedelta(minutes=15)
        
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def get_sp500_symbols(self) -> List[str]:
        """Get current S&P 500 constituents"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            return []
            
    async def get_market_indicators(self) -> Dict:
        """Get current market indicators (VIX, yield curves, etc.)"""
        try:
            indicators = {}
            
            # Get VIX
            vix = yf.Ticker("^VIX")
            indicators['vix'] = vix.info.get('regularMarketPrice')
            
            # Get Treasury yields
            treasury_10y = web.DataReader(
                "DGS10", "fred", 
                start=datetime.now() - timedelta(days=5)
            )
            treasury_2y = web.DataReader(
                "DGS2", "fred",
                start=datetime.now() - timedelta(days=5)
            )
            
            indicators['treasury_10y'] = treasury_10y.iloc[-1][0]
            indicators['treasury_2y'] = treasury_2y.iloc[-1][0]
            indicators['yield_spread'] = indicators['treasury_10y'] - indicators['treasury_2y']
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching market indicators: {e}")
            return {}
            
    async def get_sector_performance(self) -> Dict[str, float]:
        """Get sector ETF performance"""
        try:
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer': 'XLY',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Industrial': 'XLI',
                'Real Estate': 'XLRE'
            }
            
            performance = {}
            for sector, symbol in sector_etfs.items():
                etf = yf.Ticker(symbol)
                info = etf.info
                performance[sector] = info.get('regularMarketChangePercent', 0)
                
            return performance
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
            
    async def get_premarket_movers(self, min_volume: int = 100000) -> pd.DataFrame:
        """Get top pre-market movers"""
        try:
            sp500 = await self.get_sp500_symbols()
            premarket_data = []
            
            for symbol in sp500:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info.get('preMarketVolume', 0) > min_volume:
                    premarket_data.append({
                        'symbol': symbol,
                        'price': info.get('preMarketPrice'),
                        'change': info.get('preMarketChangePercent'),
                        'volume': info.get('preMarketVolume')
                    })
                    
            return pd.DataFrame(premarket_data)
            
        except Exception as e:
            logger.error(f"Error fetching pre-market movers: {e}")
            return pd.DataFrame()
            
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data"""
        try:
            returns = data['Close'].pct_change()
            return returns.dropna()
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series()

    async def get_fundamental_data(self, symbol: str) -> Dict:
        """Fetch fundamental data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE'),
                'eps': info.get('trailingEPS'),
                'revenue_growth': info.get('revenueGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'profit_margins': info.get('profitMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {} 