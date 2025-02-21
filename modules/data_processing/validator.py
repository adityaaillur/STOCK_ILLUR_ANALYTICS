import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime

class DataValidator:
    """Validates financial data quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'price': {
                'min': 0,
                'max': 1_000_000,
                'required': True
            },
            'volume': {
                'min': 0,
                'max': 10_000_000_000,
                'required': True
            },
            'market_cap': {
                'min': 0,
                'max': 1_000_000_000_000,
                'required': False
            },
            'pe_ratio': {
                'min': 0,
                'max': 1000,
                'required': False
            },
            'eps': {
                'min': -1000,
                'max': 1000,
                'required': False
            },
            'dividend_yield': {
                'min': 0,
                'max': 100,
                'required': False
            }
        }
        
    def validate_market_data(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate market data against defined rules"""
        validation_errors = {}
        
        try:
            # Check required columns
            for col, rules in self.validation_rules.items():
                if rules['required'] and col not in data.columns:
                    validation_errors[col] = ['Column is required but missing']
                    
            # Validate each column
            for col in data.columns:
                if col in self.validation_rules:
                    errors = self._validate_column(data[col], self.validation_rules[col])
                    if errors:
                        validation_errors[col] = errors
                        
            # Check date consistency
            if 'date' in data.columns:
                date_errors = self._validate_dates(data['date'])
                if date_errors:
                    validation_errors['date'] = date_errors
                    
            # Check for duplicates
            if 'date' in data.columns and 'symbol' in data.columns:
                duplicates = data.duplicated(subset=['date', 'symbol'])
                if duplicates.any():
                    validation_errors['duplicates'] = [
                        f"Found {duplicates.sum()} duplicate rows"
                    ]
                    
            return validation_errors
            
        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            return {'validation_error': [str(e)]}
    
    def validate_fundamental_data(self, data: Dict) -> Dict[str, List[str]]:
        """Validate fundamental data against defined rules"""
        validation_errors = {}
        
        try:
            for field, rules in self.validation_rules.items():
                if field in data:
                    errors = self._validate_value(data[field], rules)
                    if errors:
                        validation_errors[field] = errors
                        
            return validation_errors
            
        except Exception as e:
            logger.error(f"Error validating fundamental data: {e}")
            return {'validation_error': [str(e)]}
    
    def _validate_column(self, series: pd.Series, rules: Dict) -> List[str]:
        """Validate a pandas series against validation rules"""
        errors = []
        
        # Check for missing values
        if rules.get('required', False) and series.isna().any():
            errors.append(f"Contains {series.isna().sum()} missing values")
            
        # Check value ranges
        if 'min' in rules:
            below_min = series < rules['min']
            if below_min.any():
                errors.append(f"Contains {below_min.sum()} values below minimum {rules['min']}")
                
        if 'max' in rules:
            above_max = series > rules['max']
            if above_max.any():
                errors.append(f"Contains {above_max.sum()} values above maximum {rules['max']}")
                
        return errors
    
    def _validate_value(self, value: float, rules: Dict) -> List[str]:
        """Validate a single value against validation rules"""
        errors = []
        
        # Check for missing values
        if rules.get('required', False) and pd.isna(value):
            errors.append("Value is required but missing")
            
        # Check value ranges
        if 'min' in rules and not pd.isna(value):
            if value < rules['min']:
                errors.append(f"Value {value} is below minimum {rules['min']}")
                
        if 'max' in rules and not pd.isna(value):
            if value > rules['max']:
                errors.append(f"Value {value} is above maximum {rules['max']}")
                
        return errors
    
    def _validate_dates(self, dates: pd.Series) -> List[str]:
        """Validate date column for consistency"""
        errors = []
        
        # Check for missing dates
        if dates.isna().any():
            errors.append(f"Contains {dates.isna().sum()} missing dates")
            
        # Check for chronological order
        if not dates.is_monotonic_increasing:
            errors.append("Dates are not in chronological order")
            
        # Check for gaps in dates
        date_diff = dates.diff().dropna()
        if (date_diff > pd.Timedelta(days=7)).any():
            errors.append("Contains gaps larger than 7 days")
            
        return errors 