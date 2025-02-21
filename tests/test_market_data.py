import pytest
from modules.services.market_data import MarketDataService

@pytest.mark.asyncio
async def test_market_data_fetch():
    service = MarketDataService()
    data = await service.fetch_data("AAPL")
    assert not data.empty 