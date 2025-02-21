from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..database.models import Stock, Fundamental, TechnicalSignal, PriceData
from ..database.session import get_db
from ..schemas import (
    StockCreate, StockResponse, 
    FundamentalCreate, FundamentalResponse,
    TechnicalSignalCreate, TechnicalSignalResponse
)

router = APIRouter()

@router.get("/stocks/", response_model=List[StockResponse])
async def get_stocks(
    skip: int = 0,
    limit: int = 100,
    sector: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get list of stocks with optional filters"""
    query = db.query(Stock)
    
    if sector:
        query = query.filter(Stock.sector == sector)
    if min_market_cap:
        query = query.filter(Stock.market_cap >= min_market_cap)
        
    stocks = query.offset(skip).limit(limit).all()
    return stocks

@router.get("/stocks/{symbol}/analysis")
async def get_stock_analysis(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive analysis for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
        
    # Get latest fundamental data
    fundamental = db.query(Fundamental)\
        .filter(Fundamental.stock_id == stock.id)\
        .order_by(Fundamental.analysis_date.desc())\
        .first()
        
    # Get latest technical signals
    technical = db.query(TechnicalSignal)\
        .filter(TechnicalSignal.stock_id == stock.id)\
        .order_by(TechnicalSignal.signal_date.desc())\
        .first()
        
    # Get recent price data
    prices = db.query(PriceData)\
        .filter(
            PriceData.stock_id == stock.id,
            PriceData.date >= datetime.now() - timedelta(days=30)
        )\
        .all()
        
    return {
        "stock": stock,
        "fundamental": fundamental,
        "technical": technical,
        "recent_prices": prices
    }

@router.get("/analysis/top-picks")
async def get_top_picks(
    min_score: float = 70,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get top stock picks based on analysis"""
    top_stocks = db.query(Stock, Fundamental)\
        .join(Fundamental)\
        .filter(Fundamental.composite_score >= min_score)\
        .order_by(Fundamental.composite_score.desc())\
        .limit(limit)\
        .all()
        
    return top_stocks

@router.get("/analysis/sector-performance")
async def get_sector_performance(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get sector performance analysis"""
    # Implementation for sector performance analysis
    pass

@router.get("/technical/signals")
async def get_technical_signals(
    signal_type: Optional[str] = None,
    days: int = 1,
    db: Session = Depends(get_db)
):
    """Get stocks with specific technical signals"""
    # Implementation for technical signals query
    pass 