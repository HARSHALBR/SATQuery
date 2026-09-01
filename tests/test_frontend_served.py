from fastapi.testclient import TestClient
from backend.main import app

def test_frontend_served():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SATQuery AI" in response.text
    assert "Execute Pipeline" in response.text
