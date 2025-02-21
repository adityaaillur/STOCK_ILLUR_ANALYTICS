import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from scipy.stats import norm

class QuantitativeAnalysis:
    """
    Implements various quantitative models for stock analysis
    including DCF, Monte Carlo simulation, and Black-Scholes
    """
    
    def __init__(self):
        self.risk_free_rate = 0.04  # 4% treasury yield
        
    def calculate_dcf(self, 
                     cash_flows: List[float],
                     growth_rate: float,
                     wacc: float,
                     periods: int = 5) -> Dict[str, float]:
        """
        Calculate Discounted Cash Flow valuation
        
        Parameters:
        -----------
        cash_flows: List[float]
            Historical cash flows
        growth_rate: float
            Expected growth rate
        wacc: float
            Weighted average cost of capital
        periods: int
            Number of years to project
            
        Returns:
        --------
        Dict with DCF value and related metrics
        """
        try:
            projected_cf = []
            base_cf = cash_flows[-1]
            
            # Project future cash flows
            for i in range(periods):
                cf = base_cf * (1 + growth_rate) ** i
                projected_cf.append(cf)
                
            # Calculate terminal value
            terminal_value = (projected_cf[-1] * (1 + growth_rate)) / (wacc - growth_rate)
            
            # Calculate present value of cash flows
            present_values = [cf / (1 + wacc) ** i for i, cf in enumerate(projected_cf)]
            dcf_value = sum(present_values) + (terminal_value / (1 + wacc) ** periods)
            
            return {
                'dcf_value': dcf_value,
                'terminal_value': terminal_value,
                'projected_cash_flows': projected_cf,
                'present_values': present_values
            }
            
        except Exception as e:
            logger.error(f"DCF calculation error: {e}")
            raise

    def monte_carlo_simulation(self,
                             current_price: float,
                             volatility: float,
                             expected_return: float,
                             days: int = 252,
                             simulations: int = 10000) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo simulation for price prediction
        
        Parameters:
        -----------
        current_price: float
            Current stock price
        volatility: float
            Historical volatility
        expected_return: float
            Expected annual return
        days: int
            Number of trading days to simulate
        simulations: int
            Number of simulation runs
            
        Returns:
        --------
        Dict with simulation results
        """
        try:
            dt = 1/days
            price_paths = np.zeros((simulations, days))
            price_paths[:, 0] = current_price
            
            # Generate price paths
            for t in range(1, days):
                z = np.random.standard_normal(simulations)
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (expected_return - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * z
                )
                
            return {
                'price_paths': price_paths,
                'mean_path': np.mean(price_paths, axis=0),
                'percentile_95': np.percentile(price_paths, 95, axis=0),
                'percentile_5': np.percentile(price_paths, 5, axis=0)
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            raise

    def black_scholes(self,
                     S: float,
                     K: float,
                     T: float,
                     r: float,
                     sigma: float,
                     option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Black-Scholes option pricing
        
        Parameters:
        -----------
        S: float
            Current stock price
        K: float
            Strike price
        T: float
            Time to expiration (in years)
        r: float
            Risk-free rate
        sigma: float
            Volatility
        option_type: str
            'call' or 'put'
            
        Returns:
        --------
        Dict with option price and Greeks
        """
        try:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                delta = norm.cdf(d1)
            else:  # put
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                delta = -norm.cdf(-d1)
                
            gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = -(S*sigma*norm.pdf(d1))/(2*np.sqrt(T))
            vega = S*np.sqrt(T)*norm.pdf(d1)
            
            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            logger.error(f"Black-Scholes calculation error: {e}")
            raise

    def calculate_var(self,
                     returns: np.ndarray,
                     confidence_level: float = 0.95,
                     time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk
        
        Parameters:
        -----------
        returns: np.ndarray
            Historical returns
        confidence_level: float
            Confidence level (e.g., 0.95 for 95%)
        time_horizon: int
            Time horizon in days
            
        Returns:
        --------
        VaR value
        """
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            var_t = var * np.sqrt(time_horizon)
            return var_t
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Create instance
    quant = QuantitativeAnalysis()
    
    # Initial parameters
    current_price = 100.0
    volatility = 0.2      # 20% volatility
    expected_return = 0.1 # 10% expected return
    
    # Run Monte Carlo simulation
    results = quant.monte_carlo_simulation(
        current_price=current_price,
        volatility=volatility,
        expected_return=expected_return
    )
    
    # Get results
    final_prices = results['price_paths'][:, -1]
    mean_final_price = np.mean(final_prices)
    prob_above_start = np.mean(final_prices > current_price)
    
    print(f"Mean final price: {mean_final_price:.2f}")
    print(f"Probability final price > start: {prob_above_start:.2%}")
