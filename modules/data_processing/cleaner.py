import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime

class DataCleaner:
    """Handles data cleaning and transformation for financial data"""
    
    def __init__(self):
        self.default_values = {
            'price': 0.0,
            'volume': 0,
            'market_cap': 0.0,
            'pe_ratio': np.nan,
            'eps': np.nan,
            'dividend_yield': 0.0
        }
        
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform raw market data"""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Convert data types
            data = self._convert_data_types(data)
            
            # Remove outliers
            data = self._remove_outliers(data)
            
            # Normalize data
            data = self._normalize_data(data)
            
            # Add timestamp
            data['processed_at'] = datetime.utcnow()
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning market data: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Fill missing values with defaults
            for col, default in self.default_values.items():
                if col in data.columns:
                    data[col].fillna(default, inplace=True)
                    
            # Forward fill time series data
            if 'date' in data.columns:
                data.sort_values('date', inplace=True)
                data.ffill(inplace=True)
                
            return data
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data to appropriate types"""
        try:
            type_mapping = {
                'price': 'float64',
                'volume': 'int64',
                'market_cap': 'float64',
                'pe_ratio': 'float64',
                'eps': 'float64',
                'dividend_yield': 'float64'
            }
            
            for col, dtype in type_mapping.items():
                if col in data.columns:
                    data[col] = data[col].astype(dtype)
                    
            return data
        except Exception as e:
            logger.error(f"Error converting data types: {e}")
            return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from the data"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in self.default_values:  # Only clean specified columns
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    # Define outlier bounds
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Cap values at bounds
                    data[col] = np.where(
                        data[col] < lower_bound,
                        lower_bound,
                        np.where(
                            data[col] > upper_bound,
                            upper_bound,
                            data[col]
                        )
                    )
                    
            return data
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric data for analysis"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in self.default_values:  # Only normalize specified columns
                    # Min-max normalization
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val != min_val:  # Avoid division by zero
                        data[f'{col}_normalized'] = (data[col] - min_val) / (max_val - min_val)
                    
            return data
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data
    
    def clean_fundamental_data(self, data: Dict) -> Dict:
        """Clean and transform fundamental data"""
        try:
            cleaned_data = {}
            
            # Handle missing values
            for key, value in data.items():
                cleaned_data[key] = self.default_values.get(key, np.nan) if pd.isna(value) else value
                
            # Convert types
            cleaned_data['market_cap'] = float(cleaned_data.get('market_cap', 0))
            cleaned_data['pe_ratio'] = float(cleaned_data.get('pe_ratio', np.nan))
            cleaned_data['eps'] = float(cleaned_data.get('eps', np.nan))
            cleaned_data['dividend_yield'] = float(cleaned_data.get('dividend_yield', 0))
            
            # Add timestamp
            cleaned_data['processed_at'] = datetime.utcnow()
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning fundamental data: {e}")
            return {} 