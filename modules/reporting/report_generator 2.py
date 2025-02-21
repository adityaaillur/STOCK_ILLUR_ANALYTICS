import pandas as pd
from typing import Dict, List
from loguru import logger
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator:
    """Generates comprehensive pre-market analysis reports"""
    
    def __init__(self):
        self.report_timestamp = None
        
    def generate_premarket_report(self,
                                market_data: Dict,
                                technical_signals: Dict,
                                fundamental_data: Dict,
                                risk_metrics: Dict,
                                news_sentiment: str) -> Dict:
        """Generate complete pre-market analysis report"""
        try:
            self.report_timestamp = datetime.now()
            
            report = {
                'timestamp': self.report_timestamp.isoformat(),
                'market_summary': self._generate_market_summary(
                    market_data, news_sentiment
                ),
                'top_opportunities': self._identify_opportunities(
                    technical_signals, fundamental_data
                ),
                'risk_warnings': self._compile_risk_warnings(risk_metrics),
                'charts': self._generate_charts(market_data),
                'detailed_analysis': self._generate_detailed_analysis(
                    technical_signals, fundamental_data
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}
    
    def _generate_market_summary(self, market_data: Dict, sentiment: str) -> Dict:
        """Generate market overview section"""
        return {
            'market_sentiment': sentiment,
            'sp500_premarket': market_data.get('sp500_change', 0),
            'vix_level': market_data.get('vix', 0),
            'trending_sectors': self._get_trending_sectors(market_data),
            'major_news': market_data.get('major_news', [])
        }
    
    def _identify_opportunities(self, 
                              technical: Dict, 
                              fundamental: Dict) -> List[Dict]:
        """Identify and rank top trading opportunities"""
        opportunities = []
        
        for symbol in technical.keys():
            score = 0
            signals = []
            
            # Technical signals
            if technical[symbol].get('rsi', 0) < 30:
                score += 1
                signals.append("Oversold RSI")
            if "GOLDEN_CROSS" in technical[symbol].get('signals', []):
                score += 2
                signals.append("Golden Cross")
                
            # Fundamental factors
            fund_data = fundamental.get(symbol, {})
            if fund_data.get('pe_ratio', 100) < 15:
                score += 1
                signals.append("Low P/E")
            if fund_data.get('revenue_growth', 0) > 0.2:
                score += 1
                signals.append("Strong Growth")
                
            if score >= 2:  # Only include if multiple positive factors
                opportunities.append({
                    'symbol': symbol,
                    'score': score,
                    'signals': signals,
                    'recommendation': 'Strong Buy' if score >= 3 else 'Buy'
                })
                
        return sorted(opportunities, key=lambda x: x['score'], reverse=True)[:15]
    
    def _compile_risk_warnings(self, risk_metrics: Dict) -> List[str]:
        """Compile risk warnings and alerts"""
        warnings = []
        
        if risk_metrics.get('portfolio_beta', 0) > 1.2:
            warnings.append("High Portfolio Beta")
        if risk_metrics.get('max_correlation', 0) > 0.7:
            warnings.append("High Stock Correlation")
        for sector, exposure in risk_metrics.get('sector_exposure', {}).items():
            if exposure > 0.3:
                warnings.append(f"High {sector} Exposure")
                
        return warnings
    
    def _generate_charts(self, market_data: Dict) -> Dict:
        """Generate visualization charts"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sector Performance', 'Pre-Market Movers', 
                              'Technical Signals', 'Risk Metrics')
            )
            
            # Add sector performance chart
            sectors = market_data.get('sector_performance', {})
            fig.add_trace(
                go.Bar(x=list(sectors.keys()), 
                      y=list(sectors.values()),
                      name="Sector Returns"),
                row=1, col=1
            )
            
            # Add other charts as needed...
            
            return {
                'sector_chart': fig.to_html(),
                # Add other chart data...
            }
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {}
    
    def _generate_detailed_analysis(self, 
                                  technical: Dict, 
                                  fundamental: Dict) -> List[Dict]:
        """Generate detailed analysis for each stock"""
        analysis = []
        
        for symbol in technical.keys():
            tech_data = technical[symbol]
            fund_data = fundamental.get(symbol, {})
            
            analysis.append({
                'symbol': symbol,
                'technical_analysis': {
                    'trend': self._determine_trend(tech_data),
                    'support_resistance': self._calculate_support_resistance(tech_data),
                    'signals': tech_data.get('signals', [])
                },
                'fundamental_analysis': {
                    'valuation': self._analyze_valuation(fund_data),
                    'growth': self._analyze_growth(fund_data),
                    'health': self._analyze_financial_health(fund_data)
                }
            })
            
        return analysis 