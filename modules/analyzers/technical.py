import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class TechnicalAnalyzer:
    """Performs technical analysis on stock data"""
    
    def __init__(self):
        self.required_periods = {
            'sma': [50, 200],
            'rsi': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate technical indicators for a stock"""
        try:
            results = {'symbol': symbol}
            
            # Calculate SMAs
            for period in self.required_periods['sma']:
                sma = SMAIndicator(data['Close'], window=period)
                results[f'sma_{period}'] = sma.sma_indicator().iloc[-1]
            
            # Calculate RSI
            rsi = RSIIndicator(data['Close'], 
                             window=self.required_periods['rsi'])
            results['rsi'] = rsi.rsi().iloc[-1]
            
            # Calculate MACD
            macd = MACD(data['Close'],
                       window_fast=self.required_periods['macd']['fast'],
                       window_slow=self.required_periods['macd']['slow'],
                       window_sign=self.required_periods['macd']['signal'])
            
            results['macd_line'] = macd.macd().iloc[-1]
            results['macd_signal'] = macd.macd_signal().iloc[-1]
            results['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # Add signals
            results['signals'] = self._generate_signals(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return {}
    
    def _generate_signals(self, indicators: Dict) -> List[str]:
        """Generate trading signals based on technical indicators"""
        signals = []
        
        # RSI signals
        if indicators['rsi'] < 30:
            signals.append("RSI_OVERSOLD")
        elif indicators['rsi'] > 70:
            signals.append("RSI_OVERBOUGHT")
            
        # MACD signals
        if indicators['macd_histogram'] > 0 and indicators['macd_histogram'] > indicators['macd_histogram_prev']:
            signals.append("MACD_BULLISH")
        elif indicators['macd_histogram'] < 0 and indicators['macd_histogram'] < indicators['macd_histogram_prev']:
            signals.append("MACD_BEARISH")
            
        # Moving Average signals
        if indicators['sma_50'] > indicators['sma_200']:
            signals.append("GOLDEN_CROSS")
        elif indicators['sma_50'] < indicators['sma_200']:
            signals.append("DEATH_CROSS")
            
        return signals 