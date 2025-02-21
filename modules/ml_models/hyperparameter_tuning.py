from typing import Dict, Any
from loguru import logger
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

class HyperparameterTuner:
    """Handles hyperparameter tuning for machine learning models"""
    
    def __init__(self):
        self.param_distributions = {
            'random_forest': {
                'n_estimators': randint(50, 500),
                'max_depth': [None] + list(np.arange(3, 20)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20)
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20)
            },
            'xgboost': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5)
            },
            'lightgbm': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'num_leaves': randint(20, 100),
                'max_depth': randint(3, 10)
            },
            'catboost': {
                'iterations': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'depth': randint(3, 10)
            }
        }
        
    def tune_model(self, model, model_name: str, X, y) -> Dict[str, Any]:
        """Perform hyperparameter tuning using randomized search"""
        try:
            if model_name not in self.param_distributions:
                logger.warning(f"No parameter distribution found for {model_name}")
                return {}
                
            search = RandomizedSearchCV(
                model,
                param_distributions=self.param_distributions[model_name],
                n_iter=20,
                cv=3,
                scoring='neg_mean_squared_error',
                random_state=42
            )
            search.fit(X, y)
            
            return {
                'best_params': search.best_params_,
                'best_score': -search.best_score_,
                'best_model': search.best_estimator_
            }
            
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            return {} 