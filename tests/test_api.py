from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

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
        }
    )
    assert response.status_code == 422 # Unprocessable Entity
