import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from joblib import dump, load
import os
from .hyperparameter_tuning import HyperparameterTuner
from .explainability import ModelExplainer
from .time_series_models import TimeSeriesPredictor
import datetime
from pathlib import Path
import json

class PricePredictor:
    """
    A comprehensive price prediction system that supports multiple machine learning models,
    including traditional regression models, tree-based models, and time series models.
    
    Features:
    - Supports Random Forest, Gradient Boosting, Linear Regression, SVR, MLP, XGBoost, LightGBM, CatBoost
    - Includes ARIMA and LSTM for time series forecasting
    - Automatic hyperparameter tuning for supported models
    - Model explainability using SHAP and LIME
    - Model persistence and loading
    - Comprehensive data preparation and feature engineering
    - Detailed performance metrics and model comparison
    
    Usage:
    1. Initialize the predictor: predictor = PricePredictor()
    2. Prepare and train models: results = predictor.train_models(data)
    3. Make predictions: predictions = predictor.predict(new_data)
    4. Explain model: explanations = predictor.explain_model(model_name, data)
    5. Use time series models: ts_predictions = predictor.predict_time_series('arima', ts_data)
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the price predictor.
        
        Args:
            model_dir (str): Directory to store trained models. Defaults to 'models'.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': make_pipeline(StandardScaler(), LinearRegression()),
            'svr': make_pipeline(StandardScaler(), SVR()),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'catboost': cb.CatBoostRegressor(iterations=100, random_state=42, verbose=0),
            'arima': None,  # Will be initialized when used
            'lstm': None     # Will be initialized when used
        }
        self.best_model = None
        self.features = None
        self.target = None
        self.tuner = HyperparameterTuner()
        self.time_series_predictor = TimeSeriesPredictor()
        self.explainer = None
        self.version_format = "%Y%m%d_%H%M%S"
        self.performance_history = {}  # Track model performance over time
        self.alert_thresholds = {
            'mse': 0.1,  # 10% increase
            'mae': 0.1,  # 10% increase
            'r2': -0.1   # 10% decrease
        }
        self.alert_history = {}
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for training by selecting features, handling missing values,
        and ensuring proper data types.
        
        Args:
            data (pd.DataFrame): Raw input data containing features and target variable.
            
        Returns:
            pd.DataFrame: Cleaned and prepared data ready for training.
        """
        try:
            # Feature selection
            self.features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
                'volatility_5d', 'volatility_10d', 'volatility_20d',
                'volume_ma_5', 'volume_ma_20', 'volume_change',
                'pe_ratio', 'eps', 'dividend_yield'
            ]
            
            # Target variable
            self.target = 'close'
            
            # Filter and clean data
            data = data.dropna(subset=self.features + [self.target])
            data = data[self.features + [self.target]]
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train all available models and return performance metrics.
        
        Args:
            data (pd.DataFrame): Prepared data for training.
            
        Returns:
            Dict[str, Dict]: Dictionary containing training results for each model,
                            including performance metrics and best parameters.
        """
        try:
            # Prepare data
            data = self.prepare_data(data)
            if data is None:
                return {}
                
            # Split data
            X = data[self.features]
            y = data[self.target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize explainer
            self.explainer = ModelExplainer(X_train.values, self.features)
            
            # Train and evaluate models
            results = {}
            for model_name, model in self.models.items():
                if model_name in ['arima', 'lstm']:
                    continue  # Handle time series models separately
                    
                logger.info(f"Training {model_name}...")
                
                # Hyperparameter tuning
                if model_name in self.tuner.param_distributions:
                    tuning_result = self.tuner.tune_model(model, model_name, X_train, y_train)
                    if tuning_result:
                        model = tuning_result['best_model']
                        results[model_name] = {
                            'best_params': tuning_result['best_params'],
                            'best_score': tuning_result['best_score']
                        }
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                results[model_name].update({
                    'mse': mse,
                    'mae': mae,
                    'model': model
                })
                
                # Save model
                self._save_model(model, model_name)
                
            # Train time series models
            time_series_data = data[self.target].values
            arima_result = self.time_series_predictor.train_arima(time_series_data)
            lstm_result = self.time_series_predictor.train_lstm(time_series_data)
            
            results['arima'] = arima_result
            results['lstm'] = lstm_result
            
            # Select best model
            self.best_model = min(
                [(k, v) for k, v in results.items() if 'mse' in v],
                key=lambda x: x[1]['mse']
            )[0]
            logger.info(f"Best model: {self.best_model}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def explain_model(self, model_name: str, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain model predictions using SHAP and LIME.
        
        Args:
            model_name (str): Name of the model to explain.
            X (pd.DataFrame): Input data for explanation.
            
        Returns:
            Dict[str, Any]: Dictionary containing SHAP and LIME explanations.
        """
        try:
            if self.explainer is None:
                raise ValueError("Explainer not initialized. Train models first.")
                
            model = self._load_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
                
            # Prepare data
            data = self.prepare_data(X)
            if data is None:
                return {}
                
            X_data = data[self.features].values
            
            return {
                'shap': self.explainer.explain_shap(model, X_data),
                'lime': self.explainer.explain_lime(model, X_data, 0)  # Explain first instance
            }
            
        except Exception as e:
            logger.error(f"Error explaining model: {e}")
            return {}
    
    def predict_time_series(self, model_name: str, data: pd.Series, steps: int = 10) -> np.ndarray:
        """
        Make predictions using time series models (ARIMA or LSTM).
        
        Args:
            model_name (str): Name of the time series model ('arima' or 'lstm').
            data (pd.Series): Time series data for prediction.
            steps (int): Number of steps to predict. Defaults to 10.
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        try:
            if model_name not in ['arima', 'lstm']:
                raise ValueError("Invalid time series model")
                
            model = self._load_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
                
            if model_name == 'arima':
                return self.time_series_predictor.predict_arima(model, steps)
            else:
                return self.time_series_predictor.predict_lstm(model, data.values)
                
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            return np.array([])
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using the specified model"""
        try:
            if model_name is None:
                model_name = self.best_model
                
            model = self._load_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
                
            # Prepare data
            data = self.prepare_data(data)
            if data is None:
                return np.array([])
                
            return model.predict(data[self.features])
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def _get_versioned_path(self, model_name: str) -> Tuple[str, str]:
        """Generate versioned path for model saving"""
        version = datetime.datetime.now().strftime(self.version_format)
        base_name = f"{model_name}_{version}"
        return base_name, os.path.join(self.model_dir, base_name)
        
    def _save_model(self, model, model_name: str) -> bool:
        """Save trained model to disk with versioning"""
        try:
            base_name, path = self._get_versioned_path(model_name)
            
            if model_name in ['arima', 'lstm']:
                if model_name == 'arima':
                    model.save(f"{path}.pkl")
                else:
                    model.save(f"{path}.h5")
            else:
                dump(model, f"{path}.joblib")
                
            # Create symlink to latest version
            latest_path = os.path.join(self.model_dir, f"{model_name}_latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(base_name, latest_path)
            
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def _load_model(self, model_name: str, version: str = 'latest'):
        """Load trained model from disk with versioning"""
        try:
            if version == 'latest':
                path = os.path.join(self.model_dir, f"{model_name}_latest")
            else:
                path = os.path.join(self.model_dir, f"{model_name}_{version}")
                
            if model_name in ['arima', 'lstm']:
                if model_name == 'arima':
                    from statsmodels.tsa.arima.model import ARIMAResults
                    return ARIMAResults.load(f"{path}.pkl")
                else:
                    from tensorflow.keras.models import load_model
                    return load_model(f"{path}.h5")
            else:
                return load(f"{path}.joblib")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            if model_name is None:
                model_name = self.best_model
                
            model = self._load_model(model_name)
            if model is None:
                return {}
                
            if hasattr(model, 'feature_importances_'):
                return dict(zip(self.features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(self.features, model.coef_))
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def evaluate_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, float]:
        """Evaluate model performance on new data"""
        try:
            model = self._load_model(model_name)
            if model is None:
                return {}
                
            # Prepare data
            data = self.prepare_data(data)
            if data is None:
                return {}
                
            X = data[self.features]
            y = data[self.target]
            
            # Make predictions
            y_pred = model.predict(X)
            
            return {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': model.score(X, y)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def validate_model(self, model_name: str, version: str = 'latest') -> Dict[str, Any]:
        """
        Validate a trained model by checking:
        - Model file existence
        - Model structure
        - Compatibility with current features
        
        Args:
            model_name (str): Name of the model to validate
            version (str): Version of the model to validate. Defaults to 'latest'
            
        Returns:
            Dict[str, Any]: Validation results including status and details
        """
        try:
            # Check model existence
            if version == 'latest':
                path = os.path.join(self.model_dir, f"{model_name}_latest")
            else:
                path = os.path.join(self.model_dir, f"{model_name}_{version}")
                
            if not os.path.exists(path):
                return {
                    'status': 'error',
                    'message': f"Model {model_name} version {version} not found"
                }
                
            # Load model
            model = self._load_model(model_name, version)
            if model is None:
                return {
                    'status': 'error',
                    'message': f"Failed to load model {model_name} version {version}"
                }
                
            # Check model structure
            validation_result = {
                'status': 'success',
                'model_name': model_name,
                'version': version,
                'checks': []
            }
            
            if model_name in ['arima', 'lstm']:
                # Time series model specific checks
                if model_name == 'arima':
                    validation_result['checks'].append({
                        'check': 'arima_order',
                        'status': 'success',
                        'details': model.model.order
                    })
                else:
                    validation_result['checks'].append({
                        'check': 'lstm_layers',
                        'status': 'success',
                        'details': [layer.name for layer in model.layers]
                    })
            else:
                # Feature-based model checks
                if hasattr(model, 'feature_importances_'):
                    validation_result['checks'].append({
                        'check': 'feature_importances',
                        'status': 'success',
                        'details': len(model.feature_importances_)
                    })
                elif hasattr(model, 'coef_'):
                    validation_result['checks'].append({
                        'check': 'coefficients',
                        'status': 'success',
                        'details': len(model.coef_)
                    })
                    
            # Check feature compatibility
            if self.features is not None and model_name not in ['arima', 'lstm']:
                if hasattr(model, 'n_features_in_'):
                    if model.n_features_in_ != len(self.features):
                        validation_result['status'] = 'warning'
                        validation_result['checks'].append({
                            'check': 'feature_count',
                            'status': 'warning',
                            'details': f"Model expects {model.n_features_in_} features, current features: {len(self.features)}"
                        })
                        
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def list_model_versions(self, model_name: str) -> Dict[str, Any]:
        """
        List all available versions of a model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Dictionary containing available versions and metadata
        """
        try:
            versions = []
            for f in os.listdir(self.model_dir):
                if f.startswith(model_name + '_') and not f.endswith('_latest'):
                    version = f[len(model_name)+1:]
                    if version.count('_') == 1:  # Ensure it's a versioned file
                        versions.append({
                            'version': version,
                            'path': os.path.join(self.model_dir, f),
                            'last_modified': datetime.datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(self.model_dir, f))
                            ).strftime(self.version_format)
                        })
                        
            return {
                'model_name': model_name,
                'versions': sorted(versions, key=lambda x: x['version'], reverse=True)
            }
        except Exception as e:
            logger.error(f"Error listing versions for model {model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def rollback_model(self, model_name: str, version: str) -> bool:
        """
        Rollback to a specific version of a model.
        
        Args:
            model_name (str): Name of the model to rollback
            version (str): Version to rollback to
            
        Returns:
            bool: True if rollback was successful, False otherwise
        """
        try:
            # Verify version exists
            versions = self.list_model_versions(model_name)
            if not any(v['version'] == version for v in versions['versions']):
                logger.error(f"Version {version} not found for model {model_name}")
                return False
                
            # Remove current latest symlink
            latest_path = os.path.join(self.model_dir, f"{model_name}_latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
                
            # Create new symlink to rolled back version
            os.symlink(f"{model_name}_{version}", latest_path)
            
            logger.info(f"Successfully rolled back {model_name} to version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model {model_name}: {e}")
            return False
    
    def track_model_performance(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """
        Track model performance over time.
        
        Args:
            model_name (str): Name of the model
            metrics (Dict[str, float]): Performance metrics to track
            
        Returns:
            bool: True if tracking was successful, False otherwise
        """
        try:
            timestamp = datetime.datetime.now().strftime(self.version_format)
            
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
                
            self.performance_history[model_name].append({
                'timestamp': timestamp,
                'metrics': metrics
            })
            
            # Save performance history to disk
            self._save_performance_history()
            
            return True
        except Exception as e:
            logger.error(f"Error tracking performance for {model_name}: {e}")
            return False
    
    def get_performance_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance history for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Performance history including timestamps and metrics
        """
        try:
            if model_name not in self.performance_history:
                self._load_performance_history()
                
            return {
                'model_name': model_name,
                'history': self.performance_history.get(model_name, [])
            }
        except Exception as e:
            logger.error(f"Error getting performance history for {model_name}: {e}")
            return {}
    
    def _save_performance_history(self) -> bool:
        """Save performance history to disk"""
        try:
            path = os.path.join(self.model_dir, 'performance_history.json')
            with open(path, 'w') as f:
                json.dump(self.performance_history, f)
            return True
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
            return False
    
    def _load_performance_history(self) -> bool:
        """Load performance history from disk"""
        try:
            path = os.path.join(self.model_dir, 'performance_history.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.performance_history = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return False
    
    def evaluate_and_track(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance and track results.
        
        Args:
            model_name (str): Name of the model to evaluate
            data (pd.DataFrame): Data to use for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation results and tracking status
        """
        try:
            # Evaluate model
            metrics = self.evaluate_model(data, model_name)
            if not metrics:
                return {
                    'status': 'error',
                    'message': f"Failed to evaluate model {model_name}"
                }
                
            # Track performance
            track_result = self.track_model_performance(model_name, metrics)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'tracked': track_result
            }
        except Exception as e:
            logger.error(f"Error evaluating and tracking model {model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def check_performance_alerts(self, model_name: str) -> Dict[str, Any]:
        """
        Check for performance degradation and generate alerts.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            Dict[str, Any]: Dictionary containing alert status and details
        """
        try:
            history = self.get_performance_history(model_name)
            if not history['history'] or len(history['history']) < 2:
                return {
                    'status': 'info',
                    'message': 'Not enough history to check alerts'
                }
                
            # Get last two performance records
            current = history['history'][-1]['metrics']
            previous = history['history'][-2]['metrics']
            
            alerts = []
            for metric, threshold in self.alert_thresholds.items():
                if metric in current and metric in previous:
                    if metric == 'r2':
                        # For R2, we check for decrease
                        change = current[metric] - previous[metric]
                        if change < threshold:
                            alerts.append({
                                'metric': metric,
                                'current': current[metric],
                                'previous': previous[metric],
                                'change': change,
                                'threshold': threshold,
                                'status': 'alert'
                            })
                    else:
                        # For MSE and MAE, we check for increase
                        change = (current[metric] - previous[metric]) / previous[metric]
                        if change > threshold:
                            alerts.append({
                                'metric': metric,
                                'current': current[metric],
                                'previous': previous[metric],
                                'change': change,
                                'threshold': threshold,
                                'status': 'alert'
                            })
                            
            if alerts:
                # Store alert
                timestamp = datetime.datetime.now().strftime(self.version_format)
                if model_name not in self.alert_history:
                    self.alert_history[model_name] = []
                self.alert_history[model_name].append({
                    'timestamp': timestamp,
                    'alerts': alerts
                })
                self._save_alert_history()
                
                return {
                    'status': 'alert',
                    'model_name': model_name,
                    'alerts': alerts
                }
            else:
                return {
                    'status': 'ok',
                    'model_name': model_name,
                    'message': 'No performance alerts detected'
                }
                
        except Exception as e:
            logger.error(f"Error checking performance alerts for {model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_alert_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance alert history for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Alert history including timestamps and details
        """
        try:
            if model_name not in self.alert_history:
                self._load_alert_history()
                
            return {
                'model_name': model_name,
                'history': self.alert_history.get(model_name, [])
            }
        except Exception as e:
            logger.error(f"Error getting alert history for {model_name}: {e}")
            return {}
    
    def _save_alert_history(self) -> bool:
        """Save alert history to disk"""
        try:
            path = os.path.join(self.model_dir, 'alert_history.json')
            with open(path, 'w') as f:
                json.dump(self.alert_history, f)
            return True
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
            return False
    
    def _load_alert_history(self) -> bool:
        """Load alert history from disk"""
        try:
            path = os.path.join(self.model_dir, 'alert_history.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.alert_history = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading alert history: {e}")
            return False
    
    def evaluate_track_and_alert(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance, track results, and check for alerts.
        
        Args:
            model_name (str): Name of the model to evaluate
            data (pd.DataFrame): Data to use for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation results, tracking status, and alert status
        """
        try:
            # Evaluate and track
            eval_result = self.evaluate_and_track(model_name, data)
            if eval_result['status'] != 'success':
                return eval_result
                
            # Check for alerts
            alert_result = self.check_performance_alerts(model_name)
            
            return {
                'status': 'success',
                'metrics': eval_result['metrics'],
                'tracked': eval_result['tracked'],
                'alerts': alert_result
            }
        except Exception as e:
            logger.error(f"Error evaluating, tracking, and alerting for {model_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            } 