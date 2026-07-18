from pydantic import BaseModel, Field
import datetime
from typing import List

class PredictionRequest(BaseModel):
    name: str = Field(..., example="Maruti 800 AC")
    year: int = Field(..., ge=1990, le=datetime.datetime.now().year, example=2007)
    km_driven: int = Field(..., ge=0, example=10000)
    fuel: str = Field(..., example="Petrol")
    seller_type: str = Field(..., example="Individual")
    transmission: str = Field(..., example="Manual")
    owner: str = Field(..., example="First Owner")

class SHAPFeature(BaseModel):
    feature: str
    contribution: float

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_lower_80: float
    confidence_upper_80: float
    top_3_shap_features: List[SHAPFeature]
    negotiation_insights: List[str] = []
