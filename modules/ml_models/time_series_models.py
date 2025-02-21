import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

class TimeSeriesPredictor:
    """Handles time series specific models (ARIMA, LSTM)"""
    
    def __init__(self, look_back: int = 60):
        self.look_back = look_back
        
    def train_arima(self, series: pd.Series, order: tuple = (5, 1, 0)) -> Dict[str, Any]:
        """Train ARIMA model"""
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            return {
                'model': model_fit,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
        except Exception as e:
            logger.error(f"Error training ARIMA: {e}")
            return {}
            
    def predict_arima(self, model, steps: int) -> np.ndarray:
        """Make predictions with ARIMA"""
        try:
            return model.forecast(steps=steps)
        except Exception as e:
            logger.error(f"Error predicting with ARIMA: {e}")
            return np.array([])
            
    def create_lstm_dataset(self, data: np.ndarray) -> tuple:
        """Create dataset for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)
        
    def train_lstm(self, data: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model"""
        try:
            # Prepare data
            data = data.reshape(-1, 1)
            X, y = self.create_lstm_dataset(data)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=32, verbose=1)
            
            return {
                'model': model,
                'input_shape': X.shape[1:]
            }
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return {}
            
    def predict_lstm(self, model, data: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            data = data.reshape(-1, 1)
            X, _ = self.create_lstm_dataset(data)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with LSTM: {e}")
            return np.array([]) 