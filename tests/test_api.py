from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_api_key_unauthorized():
    response = client.post(
        "/predict",
        json={
            "name": "Maruti 800 AC",
            "year": 2007,
            "km_driven": 10000, 
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner"
        }
    )
    # Should fail without API Key
    assert response.status_code == 422 # Because Header is missing, FastAPI raises 422 Validation Error for missing header

def test_api_key_invalid():
    response = client.post(
        "/predict",
        json={
            "name": "Maruti 800 AC",
            "year": 2007,
            "km_driven": 10000, 
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner"
        },
        headers={"X-API-Key": "INVALID-KEY"}
    )
    # Should fail with 401 Unauthorized
    assert response.status_code == 401
    
def test_predict_validation_error():
    # Negative km_driven should be rejected by Pydantic schema validation
    response = client.post(
        "/predict",
        json={
            "name": "Maruti 800 AC",
            "year": 2007,
            "km_driven": -1000, 
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner"
        },
        headers={"X-API-Key": "AUTOVAL-DEMO-KEY"}
    )
    assert response.status_code == 422 # Unprocessable Entity
