from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ==========================
# Test Cases for the API
# ==========================

# Test the /predict endpoint with a normal message
def test_normal_message():
    response = client.post(
        "/predict",
        json={"text": "Hey, are we meeting tomorrow?"}
    )
    
    assert response.status_code == 200
    assert response.json()["prediction"] in ["ham", "spam"]
    assert 0 <= response.json()["spam_probability"] <= 1

# Test the /predict endpoint with an empty message
def test_empty_message():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    
    assert response.status_code == 422

# Test the /predict endpoint with whitespace-only message
def test_whitespace_message():
    response = client.post(
        "/predict",
        json={"text": "   "}
    )
    
    assert response.status_code == 422

# Test the /predict endpoint with a very long message
def test_long_message():
    long_text = "This is a very long message. " * 100  # 3000 characters
    response = client.post(
        "/predict",
        json={"text": long_text}
    )
    
    assert response.status_code == 422

# test the /predict endpoint with missing message field
def test_missing_message_field():
    response = client.post(
        "/predict",
        json={}
    )
    
    assert response.status_code == 422

# Test the /predict endpoint with an invalid input type
def test_invalid_input_type():
    response = client.post(
        "/predict",
        json={"text": ["hello", "world"]}
    )
    
    assert response.status_code == 422