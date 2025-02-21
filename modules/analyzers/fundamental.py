import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger

class FundamentalAnalyzer:
    """Analyzes fundamental data to identify strong investment candidates"""
    
    def __init__(self):
        self.pe_threshold = 30
        self.growth_threshold = 0.10  # 10% growth
        self.min_market_cap = 1e9    # $1B minimum
        self.min_profit_margin = 0.10 # 10% margin
        
    def analyze_stock(self, fundamental_data: Dict) -> Dict[str, float]:
        """Analyze fundamental metrics for a single stock"""
        try:
            scores = {}
            
            # Valuation Score (0-100)
            pe_ratio = fundamental_data.get('pe_ratio', float('inf'))
            if pe_ratio > 0:  # Ignore negative P/E
                scores['valuation'] = min(100, (self.pe_threshold / pe_ratio) * 100)
            else:
                scores['valuation'] = 0
                
            # Growth Score
            revenue_growth = fundamental_data.get('revenue_growth', 0)
            scores['growth'] = min(100, (revenue_growth / self.growth_threshold) * 100)
            
            # Financial Health Score
            debt_to_equity = fundamental_data.get('debt_to_equity', float('inf'))
            current_ratio = fundamental_data.get('current_ratio', 0)
            scores['financial_health'] = self._calculate_health_score(debt_to_equity, current_ratio)
            
            # Profitability Score
            profit_margin = fundamental_data.get('profit_margins', 0)
            roe = fundamental_data.get('return_on_equity', 0)
            scores['profitability'] = self._calculate_profitability_score(profit_margin, roe)
            
            # Calculate composite score
            scores['composite'] = np.mean([
                scores['valuation'],
                scores['growth'],
                scores['financial_health'],
                scores['profitability']
            ])
            
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {e}")
            return {}
    
    def screen_stocks(self, stocks_data: List[Dict]) -> pd.DataFrame:
        """Screen stocks based on fundamental criteria"""
        try:
            screened_stocks = []
            
            for stock_data in stocks_data:
                if self._passes_screening(stock_data):
                    scores = self.analyze_stock(stock_data)
                    stock_data.update(scores)
                    screened_stocks.append(stock_data)
            
            # Convert to DataFrame and sort by composite score
            df = pd.DataFrame(screened_stocks)
            if not df.empty:
                df = df.sort_values('composite', ascending=False)
                
            return df
            
        except Exception as e:
            logger.error(f"Error screening stocks: {e}")
            return pd.DataFrame()
    
    def _calculate_health_score(self, debt_to_equity: float, current_ratio: float) -> float:
        """Calculate financial health score"""
        try:
            # Debt to Equity score (lower is better)
            if debt_to_equity <= 0:
                de_score = 100
            else:
                de_score = max(0, 100 - (debt_to_equity * 20))
                
            # Current Ratio score (higher is better)
            cr_score = min(100, current_ratio * 50)
            
            return np.mean([de_score, cr_score])
            
        except Exception:
            return 0
    
    def _calculate_profitability_score(self, profit_margin: float, roe: float) -> float:
        """Calculate profitability score"""
        try:
            # Profit margin score
            margin_score = min(100, (profit_margin / self.min_profit_margin) * 100)
            
            # ROE score
            roe_score = min(100, roe * 100)
            
            return np.mean([margin_score, roe_score])
            
        except Exception:
            return 0
    
    def _passes_screening(self, stock_data: Dict) -> bool:
        """Check if stock passes basic screening criteria"""
        try:
            # Market cap check
            if stock_data.get('market_cap', 0) < self.min_market_cap:
                return False
                
            # Positive earnings check
            if stock_data.get('eps', 0) <= 0:
                return False
                
            # Minimum profit margin check
            if stock_data.get('profit_margins', 0) < self.min_profit_margin:
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_investment_recommendations(self, 
                                    screened_stocks: pd.DataFrame,
                                    min_score: float = 70) -> List[Dict]:
        """Generate investment recommendations based on fundamental analysis"""
        try:
            recommendations = []
            
            for _, stock in screened_stocks.iterrows():
                if stock['composite'] >= min_score:
                    rec = {
                        'symbol': stock['symbol'],
                        'score': stock['composite'],
                        'strength': self._get_strength_factors(stock),
                        'risks': self._get_risk_factors(stock),
                        'recommendation': self._get_recommendation(stock)
                    }
                    recommendations.append(rec)
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _get_strength_factors(self, stock: pd.Series) -> List[str]:
        """Identify key strength factors"""
        strengths = []
        
        if stock['valuation'] >= 80:
            strengths.append("Attractive Valuation")
        if stock['growth'] >= 80:
            strengths.append("Strong Growth")
        if stock['financial_health'] >= 80:
            strengths.append("Solid Financial Health")
        if stock['profitability'] >= 80:
            strengths.append("High Profitability")
            
        return strengths
    
    def _get_risk_factors(self, stock: pd.Series) -> List[str]:
        """Identify key risk factors"""
        risks = []
        
        if stock['valuation'] <= 40:
            risks.append("High Valuation")
        if stock['growth'] <= 40:
            risks.append("Weak Growth")
        if stock['financial_health'] <= 40:
            risks.append("Financial Health Concerns")
        if stock['profitability'] <= 40:
            risks.append("Low Profitability")
            
        return risks
    
    def _get_recommendation(self, stock: pd.Series) -> str:
        """Generate recommendation based on scores"""
        score = stock['composite']
        
        if score >= 80:
            return "Strong Buy"
        elif score >= 60:
            return "Buy"
        elif score >= 40:
            return "Hold"
        else:
            return "Sell" 