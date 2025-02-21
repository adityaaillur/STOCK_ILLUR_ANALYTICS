import shap
import lime
import lime.lime_tabular
from typing import Dict, Any
import numpy as np
from loguru import logger

class ModelExplainer:
    """Provides model explainability using SHAP and LIME"""
    
    def __init__(self, X_train, feature_names):
        self.explainer_shap = shap.Explainer()
        self.explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression'
        )
        
    def explain_shap(self, model, X) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        try:
            shap_values = self.explainer_shap(model, X)
            return {
                'shap_values': shap_values.values,
                'base_value': shap_values.base_values,
                'data': shap_values.data
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {}
            
    def explain_lime(self, model, X, instance_index: int) -> Dict[str, Any]:
        """Generate LIME explanation for a specific instance"""
        try:
            exp = self.explainer_lime.explain_instance(
                X[instance_index],
                model.predict,
                num_features=10
            )
            return {
                'explanation': exp.as_list(),
                'prediction': exp.predicted_value
            }
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {} 