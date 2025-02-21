from typing import Dict, Optional, List
from loguru import logger
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import pandas as pd

from ..database.models import DataQualityHistory
from .quality_metrics import DataQualityMetrics

class QualityTracker:
    """Tracks and stores historical data quality metrics"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.metrics_calculator = DataQualityMetrics()
        
    def track_quality(self, data: pd.DataFrame, dataset_name: str) -> Optional[DataQualityHistory]:
        """Track and store data quality metrics"""
        try:
            # Calculate metrics
            report = self.metrics_calculator.generate_quality_report(data)
            
            # Create history record
            quality_record = DataQualityHistory(
                dataset_name=dataset_name,
                metrics=report['metrics'],
                issues=report['issues'],
                overall_score=report['summary']['overall_quality_score']
            )
            
            # Store in database
            self.db.add(quality_record)
            self.db.commit()
            
            logger.info(f"Tracked quality metrics for {dataset_name}")
            return quality_record
            
        except Exception as e:
            logger.error(f"Error tracking quality metrics: {e}")
            self.db.rollback()
            return None
    
    def get_quality_history(self, 
                          dataset_name: str,
                          days: int = 30) -> List[DataQualityHistory]:
        """Get quality history for a dataset"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return self.db.query(DataQualityHistory)\
                .filter(DataQualityHistory.dataset_name == dataset_name)\
                .filter(DataQualityHistory.timestamp >= cutoff_date)\
                .order_by(DataQualityHistory.timestamp.desc())\
                .all()
        except Exception as e:
            logger.error(f"Error getting quality history: {e}")
            return []
    
    def get_quality_trend(self, 
                        dataset_name: str,
                        days: int = 30) -> Dict[str, List[float]]:
        """Get quality trend data for visualization"""
        try:
            history = self.get_quality_history(dataset_name, days)
            
            trend_data = {
                'timestamps': [],
                'overall_scores': [],
                'completeness': [],
                'accuracy': [],
                'consistency': [],
                'timeliness': [],
                'uniqueness': []
            }
            
            for record in history:
                trend_data['timestamps'].append(record.timestamp)
                trend_data['overall_scores'].append(record.overall_score)
                trend_data['completeness'].append(record.metrics.get('completeness', 0))
                trend_data['accuracy'].append(record.metrics.get('accuracy', 0))
                trend_data['consistency'].append(record.metrics.get('consistency', 0))
                trend_data['timeliness'].append(record.metrics.get('timeliness', 0))
                trend_data['uniqueness'].append(record.metrics.get('uniqueness', 0))
                
            return trend_data
            
        except Exception as e:
            logger.error(f"Error getting quality trend: {e}")
            return {} 