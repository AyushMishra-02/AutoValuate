import os
import pandas as pd
from pydantic import BaseModel, ValidationError, Field, validator
from typing import List, Optional
import datetime

class CarListing(BaseModel):
    name: str
    year: int = Field(ge=1990, le=datetime.datetime.now().year)
    selling_price: float = Field(gt=0)
    km_driven: int = Field(ge=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str

    @validator('fuel')
    def validate_fuel(cls, v):
        allowed = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
        if v not in allowed:
            raise ValueError(f"Invalid fuel type: {v}")
        return v

def validate_and_clean_data(raw_filepath, processed_filepath):
    print(f"Loading raw data from {raw_filepath}...")
    df = pd.read_csv(raw_filepath)
    print(f"Initial row count: {len(df)}")
    
    valid_rows = []
    invalid_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Validate using Pydantic
            validated_car = CarListing(**row.to_dict())
            valid_rows.append(validated_car.dict())
        except ValidationError as e:
            invalid_count += 1
            # In a real pipeline, we might log these to a dead-letter queue
            pass
            
    print(f"Validation complete. Dropped {invalid_count} invalid rows.")
    
    # Save processed data
    processed_df = pd.DataFrame(valid_rows)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    processed_df.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to {processed_filepath}. Final count: {len(processed_df)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, "data", "car_data.csv") # Treating the fetched one as raw
    processed_path = os.path.join(base_dir, "data", "processed", "car_data_clean.csv")
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} does not exist. Run data_fetch.py first.")
    else:
        validate_and_clean_data(raw_path, processed_path)
