from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_dir, 'data', 'prediction_logs.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Request Payload
    name = Column(String, index=True)
    year = Column(Integer)
    km_driven = Column(Integer)
    fuel = Column(String)
    seller_type = Column(String)
    transmission = Column(String)
    owner = Column(String)
    
    # Model Predictions
    predicted_price = Column(Float)
    lower_bound_80 = Column(Float)
    upper_bound_80 = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
