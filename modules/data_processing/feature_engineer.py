import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from datetime import timedelta

class FeatureEngineer:
    """Creates new features from cleaned financial data"""
    
    def __init__(self):
        self.technical_windows = [5, 10, 20, 50, 200]  # Common technical analysis periods
        self.volume_windows = [5, 20]  # Volume analysis periods
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical and fundamental features"""
        try:
            if 'date' in data.columns:
                data = data.sort_values('date')
                
            # Price-based features
            data = self._add_price_features(data)
            
            # Volume-based features
            data = self._add_volume_features(data)
            
            # Momentum features
            data = self._add_momentum_features(data)
            
            # Volatility features
            data = self._add_volatility_features(data)
            
            # Fundamental features
            data = self._add_fundamental_features(data)
            
            # Date-based features
            data = self._add_date_features(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return pd.DataFrame()
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators"""
        try:
            # Moving averages
            for window in self.technical_windows:
                data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
                data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()
                
            # Price changes
            data['daily_return'] = data['close'].pct_change()
            for window in [1, 3, 5, 10]:
                data[f'return_{window}d'] = data['close'].pct_change(window)
                
            # Support/resistance levels
            data['support'] = data['low'].rolling(window=20).min()
            data['resistance'] = data['high'].rolling(window=20).max()
            
            return data
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            # Volume moving averages
            for window in self.volume_windows:
                data[f'volume_ma_{window}'] = data['volume'].rolling(window=window).mean()
                
            # Volume changes
            data['volume_change'] = data['volume'].pct_change()
            for window in [1, 3, 5]:
                data[f'volume_change_{window}d'] = data['volume'].pct_change(window)
                
            # Volume-price relationship
            data['volume_price_trend'] = data['volume'] * data['close'].pct_change()
            
            return data
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            
            # Stochastic Oscillator
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
            
            return data
        except Exception as e:
            logger.error(f"Error adding momentum features: {e}")
            return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        try:
            # Historical volatility
            for window in [5, 10, 20]:
                data[f'volatility_{window}d'] = data['close'].pct_change().rolling(window).std() * np.sqrt(window)
                
            # Average True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = tr.rolling(window=14).mean()
            
            return data
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return data
    
    def _add_fundamental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental analysis features"""
        try:
            # Valuation ratios
            if 'pe_ratio' in data.columns:
                data['pe_ratio_change'] = data['pe_ratio'].pct_change()
                data['pe_ratio_ma_20'] = data['pe_ratio'].rolling(window=20).mean()
                
            if 'eps' in data.columns:
                data['eps_growth'] = data['eps'].pct_change()
                
            if 'dividend_yield' in data.columns:
                data['dividend_yield_ma_20'] = data['dividend_yield'].rolling(window=20).mean()
                
            return data
        except Exception as e:
            logger.error(f"Error adding fundamental features: {e}")
            return data
    
    def _add_date_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            if 'date' in data.columns:
                data['day_of_week'] = data['date'].dt.dayofweek
                data['month'] = data['date'].dt.month
                data['quarter'] = data['date'].dt.quarter
                data['year'] = data['date'].dt.year
                data['is_month_end'] = data['date'].dt.is_month_end
                data['is_quarter_end'] = data['date'].dt.is_quarter_end
                
            return data
        except Exception as e:
            logger.error(f"Error adding date features: {e}")
            return data 