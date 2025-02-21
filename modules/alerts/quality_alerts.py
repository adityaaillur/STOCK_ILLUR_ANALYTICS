from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from ..config import settings
from ..data_processing.quality_tracker import QualityTracker

class QualityAlertSystem:
    """Monitors data quality and sends alerts when thresholds are breached"""
    
    def __init__(self, quality_tracker: QualityTracker):
        self.quality_tracker = quality_tracker
        self.last_alert_time = {}
        
    def check_quality_alerts(self, dataset_name: str) -> Optional[Dict]:
        """Check for quality alerts and send notifications"""
        try:
            # Get latest quality metrics
            history = self.quality_tracker.get_quality_history(dataset_name, days=1)
            if not history:
                return None
                
            latest_metrics = history[0].metrics
            overall_score = history[0].overall_score
            
            # Check for breaches
            alerts = self._detect_quality_breaches(
                latest_metrics,
                overall_score,
                dataset_name
            )
            
            if alerts:
                self._send_alerts(alerts)
                return alerts
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking quality alerts: {e}")
            return None
    
    def _detect_quality_breaches(self, 
                               metrics: Dict[str, float],
                               overall_score: float,
                               dataset_name: str) -> Dict[str, Dict]:
        """Detect quality metric breaches"""
        alerts = {}
        
        # Check overall score
        if overall_score < settings.QUALITY_ALERT_THRESHOLDS['overall_score']:
            alerts['overall_score'] = {
                'metric': 'overall_score',
                'value': overall_score,
                'threshold': settings.QUALITY_ALERT_THRESHOLDS['overall_score'],
                'dataset': dataset_name
            }
            
        # Check individual metrics
        for metric, threshold in settings.QUALITY_ALERT_THRESHOLDS.items():
            if metric != 'overall_score' and metric in metrics:
                if metrics[metric] < threshold:
                    alerts[metric] = {
                        'metric': metric,
                        'value': metrics[metric],
                        'threshold': threshold,
                        'dataset': dataset_name
                    }
                    
        return alerts
    
    def _send_alerts(self, alerts: Dict[str, Dict]) -> bool:
        """Send alerts through configured channels"""
        try:
            for alert in alerts.values():
                message = self._format_alert_message(alert)
                
                if 'email' in settings.ALERT_NOTIFICATION_CHANNELS:
                    self._send_email_alert(message)
                    
                if 'slack' in settings.ALERT_NOTIFICATION_CHANNELS:
                    self._send_slack_alert(message)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
            return False
    
    def _format_alert_message(self, alert: Dict) -> str:
        """Format alert message"""
        return (
            f"🚨 Data Quality Alert 🚨\n"
            f"Dataset: {alert['dataset']}\n"
            f"Metric: {alert['metric']}\n"
            f"Value: {alert['value']:.2f} (Threshold: {alert['threshold']})\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def _send_email_alert(self, message: str) -> bool:
        """Send alert via email"""
        try:
            # Implement email sending logic
            logger.info(f"Sending email alert: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def _send_slack_alert(self, message: str) -> bool:
        """Send alert via Slack"""
        try:
            # Implement Slack sending logic
            logger.info(f"Sending Slack alert: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False 