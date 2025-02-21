from sqlalchemy import Column, Integer, String, DateTime, JSON
from database.session import Base

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    activity_type = Column(String)
    endpoint = Column(String)
    metadata = Column(JSON)
    timestamp = Column(DateTime) 