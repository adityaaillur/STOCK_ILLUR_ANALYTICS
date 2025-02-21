from fastapi import APIRouter, Depends
from typing import List
from modules.database.models import MarketData
from modules.database.session import get_db

router = APIRouter(prefix="/market-data")

@router.get("/{symbol}")
async def get_market_data(symbol: str, db = Depends(get_db)):
    return {"symbol": symbol}

@router.post("/analyze")
async def analyze_market_data(symbol: str, analysis_type: str):
    return {"symbol": symbol, "analysis": analysis_type} 