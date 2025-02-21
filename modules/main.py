import asyncio
from typing import Dict, List
from loguru import logger
from datetime import datetime
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api.routes import router as api_router
from .middleware.error_handler import ErrorHandlerMiddleware
from sqlalchemy.orm import Session
from database.session import get_db
from data_processing.quality_tracker import QualityTracker
from alerts.quality_alerts import QualityAlertSystem

from data_collectors.market_data import MarketDataCollector
from data_collectors.news_scraper import NewsCollector
from analyzers.technical import TechnicalAnalyzer
from modules.risk_management.risk_analyzer import RiskAnalyzer
from modules.portfolio.optimizer import PortfolioOptimizer
from modules.reporting.report_generator import ReportGenerator
from analyzers.fundamental import FundamentalAnalyzer

app = FastAPI(
    title="Stock Analysis API",
    description="API for stock market analysis and portfolio management",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)

# Add middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

class StockAnalysisApp:
    """Main application orchestrator for stock analysis"""
    
    def __init__(self):
        self.db = get_db()
        self.quality_tracker = QualityTracker(self.db)
        self.alert_system = QualityAlertSystem(self.quality_tracker)
        self.market_data = MarketDataCollector()
        self.news_collector = NewsCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.report_generator = ReportGenerator()
        
    async def run_premarket_analysis(self) -> Dict:
        """Run complete pre-market analysis workflow"""
        try:
            logger.info("Starting pre-market analysis...")
            
            # 1. Collect market data and news
            market_data = await self.market_data.get_premarket_data(symbols=self.get_sp500_symbols())
            news = await self.news_collector.get_market_news()
            sentiment = self.news_collector.analyze_market_sentiment(news)
            
            # 2. Technical Analysis
            technical_signals = {}
            for symbol in market_data['symbol'].unique():
                stock_data = await self.market_data.get_historical_data(symbol)
                technical_signals[symbol] = self.technical_analyzer.analyze(stock_data, symbol)
            
            # 3. Fundamental Analysis
            fundamental_data = {}
            for symbol in market_data['symbol'].unique():
                fundamental_data[symbol] = await self.market_data.get_fundamental_data(symbol)
                data = await self.market_data.get_fundamental_data(symbol)
                fundamental_data[symbol] = self.fundamental_analyzer.analyze(data)
                
            # 4. Risk Analysis
            portfolio_risk = self.risk_analyzer.calculate_portfolio_risk(
                market_data, self.get_historical_returns()
            )
            
            # 5. Portfolio Optimization
            returns = self.get_historical_returns()
            optimal_portfolio = self.portfolio_optimizer.optimize_portfolio(returns)
            
            # 6. Generate Report
            report = self.report_generator.generate_premarket_report(
                market_data=market_data.to_dict(),
                technical_signals=technical_signals,
                fundamental_data=fundamental_data,
                risk_metrics=portfolio_risk,
                news_sentiment=sentiment
            )
            
            # Track data quality
            self.quality_tracker.track_quality(market_data, 'market_data')
            self.quality_tracker.track_quality(pd.DataFrame(fundamental_data), 'fundamental_data')
            
            # Check for quality alerts
            self.alert_system.check_quality_alerts('market_data')
            self.alert_system.check_quality_alerts('fundamental_data')
            
            logger.info("Pre-market analysis completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error in pre-market analysis: {e}")
            return {}
    
    def get_sp500_symbols(self) -> List[str]:
        """Get list of S&P 500 symbols"""
        # Implement S&P 500 symbols retrieval
        pass
    
    def get_historical_returns(self) -> pd.DataFrame:
        """Get historical returns for analysis"""
        # Implement historical returns retrieval
        pass

if __name__ == "__main__":
    app = StockAnalysisApp()
    asyncio.run(app.run_premarket_analysis()) 