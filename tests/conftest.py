import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from modules.database.models import Base
from modules.database.session import get_db

@pytest.fixture(scope="session")
def test_db():
    SQLALCHEMY_DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close() 