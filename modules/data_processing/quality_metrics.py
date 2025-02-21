import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime
from validators.data_validator import DataValidator

class DataQualityMetrics:
    """Calculates and tracks data quality metrics"""
    
    def __init__(self):
        self.metric_definitions = {
            'completeness': {
                'description': 'Percentage of non-missing values',
                'calculation': self._calculate_completeness
            },
            'accuracy': {
                'description': 'Percentage of values within expected ranges',
                'calculation': self._calculate_accuracy
            },
            'consistency': {
                'description': 'Percentage of consistent records',
                'calculation': self._calculate_consistency
            },
            'timeliness': {
                'description': 'Percentage of up-to-date records',
                'calculation': self._calculate_timeliness
            },
            'uniqueness': {
                'description': 'Percentage of unique records',
                'calculation': self._calculate_uniqueness
            }
        }
        
    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all data quality metrics"""
        try:
            metrics = {}
            
            for metric_name, definition in self.metric_definitions.items():
                metrics[metric_name] = definition['calculation'](data)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {e}")
            return {}
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate completeness metric"""
        try:
            total_cells = data.size
            missing_cells = data.isna().sum().sum()
            return (total_cells - missing_cells) / total_cells
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.0
    
    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate accuracy metric"""
        try:
            validator = DataValidator()
            validation_errors = validator.validate_market_data(data)
            
            total_values = data.size
            error_count = sum(len(errors) for errors in validation_errors.values())
            
            return (total_values - error_count) / total_values
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency metric"""
        try:
            # Check for consistent data types
            type_consistency = all(data[col].apply(type).nunique() == 1 for col in data.columns)
            
            # Check for consistent date ranges
            if 'date' in data.columns:
                date_diff = data['date'].diff().dropna()
                date_consistency = (date_diff <= pd.Timedelta(days=7)).all()
            else:
                date_consistency = True
                
            return float(type_consistency and date_consistency)
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return 0.0
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calculate timeliness metric"""
        try:
            if 'date' in data.columns:
                latest_date = data['date'].max()
                days_since_last = (datetime.now() - latest_date).days
                return max(0, 1 - (days_since_last / 7))  # 1 if within 7 days, 0 if older
            return 1.0  # If no date column, assume data is timely
        except Exception as e:
            logger.error(f"Error calculating timeliness: {e}")
            return 0.0
    
    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness metric"""
        try:
            if 'date' in data.columns and 'symbol' in data.columns:
                total_rows = len(data)
                unique_rows = data.drop_duplicates(subset=['date', 'symbol']).shape[0]
                return unique_rows / total_rows
            return 1.0  # If no key columns, assume all rows are unique
        except Exception as e:
            logger.error(f"Error calculating uniqueness: {e}")
            return 0.0
    
    def generate_quality_report(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        try:
            metrics = self.calculate_metrics(data)
            
            report = {
                'summary': {
                    'total_records': len(data),
                    'total_columns': len(data.columns),
                    'overall_quality_score': np.mean(list(metrics.values()))
                },
                'metrics': metrics,
                'issues': self._identify_quality_issues(data)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {}
    
    def _identify_quality_issues(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify specific data quality issues"""
        issues = {}
        
        # Missing values
        missing_values = data.isna().sum()
        if missing_values.any():
            issues['missing_values'] = [
                f"{col}: {count} missing" 
                for col, count in missing_values.items() 
                if count > 0
            ]
            
        # Outliers
        validator = DataValidator()
        validation_errors = validator.validate_market_data(data)
        if validation_errors:
            issues['validation_errors'] = [
                f"{col}: {', '.join(errors)}"
                for col, errors in validation_errors.items()
            ]
            
        # Duplicates
        if 'date' in data.columns and 'symbol' in data.columns:
            duplicates = data.duplicated(subset=['date', 'symbol'])
            if duplicates.any():
                issues['duplicates'] = [
                    f"Found {duplicates.sum()} duplicate rows"
                ]
                
        return issues 