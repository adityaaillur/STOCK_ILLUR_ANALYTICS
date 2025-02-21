import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from loguru import logger

class PortfolioOptimizer:
    """Optimizes portfolio allocation using modern portfolio theory"""
    
    def __init__(self):
        self.min_weight = 0.02  # 2% minimum position size
        self.max_weight = 0.20  # 20% maximum position size
        
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         risk_free_rate: float = 0.04,
                         target_volatility: float = None) -> Dict[str, np.ndarray]:
        """
        Optimize portfolio weights using Sharpe Ratio maximization
        or minimum variance with target return
        """
        try:
            n_assets = returns.shape[1]
            
            # Calculate mean returns and covariance
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define optimization constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            
            # Add position size constraints
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
            
            # Define objective function (negative Sharpe Ratio)
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
                return -sharpe_ratio
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Run optimization
            result = minimize(objective,
                            initial_weights,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * optimal_weights)
            portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {}
            
    def rebalance_portfolio(self,
                          current_weights: np.ndarray,
                          target_weights: np.ndarray,
                          threshold: float = 0.05) -> Dict[str, np.ndarray]:
        """Generate rebalancing trades when allocation deviates from targets"""
        try:
            deviation = np.abs(current_weights - target_weights)
            trades_needed = deviation > threshold
            
            rebalance_trades = np.zeros_like(current_weights)
            rebalance_trades[trades_needed] = target_weights[trades_needed] - current_weights[trades_needed]
            
            return {
                'trades': rebalance_trades,
                'total_deviation': deviation.sum(),
                'positions_to_trade': np.sum(trades_needed)
            }
            
        except Exception as e:
            logger.error(f"Rebalancing calculation error: {e}")
            return {}

    def optimize_portfolio(self, returns):
        # Implement portfolio optimization logic
        pass 