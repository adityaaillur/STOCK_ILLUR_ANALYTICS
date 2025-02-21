from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class StockBase(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None

class StockCreate(StockBase):
    pass

class StockResponse(StockBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class FundamentalBase(BaseModel):
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    dividend_yield: Optional[float] = None
    composite_score: Optional[float] = None
    recommendation: Optional[str] = None

class FundamentalCreate(FundamentalBase):
    stock_id: int

class FundamentalResponse(FundamentalBase):
    id: int
    stock_id: int
    analysis_date: datetime
    
    class Config:
        orm_mode = True

class TechnicalSignalBase(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    signals: Optional[List[str]] = None

class TechnicalSignalCreate(TechnicalSignalBase):
    stock_id: int

class TechnicalSignalResponse(TechnicalSignalBase):
    id: int
    stock_id: int
    signal_date: datetime
    
    class Config:
        orm_mode = True

class PriceDataBase(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class PriceDataCreate(PriceDataBase):
    stock_id: int

class PriceDataResponse(PriceDataBase):
    id: int
    stock_id: int
    
    class Config:
        orm_mode = True 