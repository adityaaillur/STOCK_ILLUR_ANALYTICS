import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def construct_portfolio(data):
    """
    Construct a risk-adjusted portfolio.
    For demonstration, select the top 3 stocks by the dummy "Valuation" column.
    """
    try:
        portfolio = data.sort_values("Valuation", ascending=False).head(3)
        logger.info("Portfolio constructed.")
        return portfolio
    except Exception as e:
        logger.error(f"Error constructing portfolio: {e}")
        return pd.DataFrame()
