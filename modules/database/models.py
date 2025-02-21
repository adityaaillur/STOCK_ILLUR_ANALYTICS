from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Stock(Base):
    """Stock basic information"""
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Relationships
    fundamentals = relationship("Fundamental", back_populates="stock")
    technical_signals = relationship("TechnicalSignal", back_populates="stock")
    price_data = relationship("PriceData", back_populates="stock")

class Fundamental(Base):
    """Fundamental analysis data"""
    __tablename__ = 'fundamentals'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'))
    pe_ratio = Column(Float)
    eps = Column(Float)
    revenue_growth = Column(Float)
    profit_margin = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    return_on_equity = Column(Float)
    dividend_yield = Column(Float)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    composite_score = Column(Float)
    recommendation = Column(String)
    
    # Relationship
    stock = relationship("Stock", back_populates="fundamentals")

class TechnicalSignal(Base):
    """Technical analysis signals"""
    __tablename__ = 'technical_signals'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'))
    signal_date = Column(DateTime, default=datetime.utcnow)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    signals = Column(JSON)  # Store list of signals as JSON
    
    # Relationship
    stock = relationship("Stock", back_populates="technical_signals")

class PriceData(Base):
    """Historical price data"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'))
    date = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    # Relationship
    stock = relationship("Stock", back_populates="price_data")

class DataQualityHistory(Base):
    """Tracks historical data quality metrics"""
    __tablename__ = 'data_quality_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    dataset_name = Column(String)
    metrics = Column(JSON)
    issues = Column(JSON)
    overall_score = Column(Float)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

class MarketData(Base):
    __tablename__ = "market_data"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

class Analysis(Base):
    __tablename__ = "analysis"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    timestamp = Column(DateTime)
    analysis_type = Column(String)
    results = Column(JSON) 