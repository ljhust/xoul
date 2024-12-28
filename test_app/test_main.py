import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app, get_db
from app.database import Base
from app.models import Comparison

# Setup an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./comparison.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the database dependency with the test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Apply the override
app.dependency_overrides[get_db] = override_get_db

# Create the test database
Base.metadata.create_all(bind=engine)

# Initialize the test client
client = TestClient(app)


@pytest.fixture(scope="function")
def db_session():
    """Fixture to reset the database before each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_compare_endpoint(db_session):
    """Test the /compare endpoint."""
    response = client.post("/compare", json={"text": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["input_text"] == "What is the capital of France?"
    assert "response_model_1" in data
    assert "response_model_2" in data
    assert data["votes_model_1"] == 0
    assert data["votes_model_2"] == 0


def test_vote_endpoint(db_session):
    """Test voting for a preferred model."""
    # First, create a comparison
    response = client.post("/compare", json={"text": "What is the capital of France?"})
    assert response.status_code == 200
    comparison_id = response.json()["id"]

    # Vote for model_1
    vote_response = client.post(f"/vote/{comparison_id}/model_1")
    assert vote_response.status_code == 200
    assert vote_response.json() == {"message": "Vote recorded successfully."}

    # Ensure the vote was counted
    response = client.post("/compare", json={"text": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["votes_model_1"] == 1
    assert data["votes_model_2"] == 0


def test_invalid_vote(db_session):
    """Test voting with invalid model choice."""
    # First, create a comparison
    response = client.post("/compare", json={"text": "What is the capital of France?"})
    assert response.status_code == 200
    comparison_id = response.json()["id"]

    # Attempt an invalid vote
    vote_response = client.post(f"/vote/{comparison_id}/invalid_model")
    assert vote_response.status_code == 400
    assert vote_response.json() == {
        "detail": "Invalid model choice. Must be 'model_1' or 'model_2'."
    }


def test_statistics_endpoint(db_session):
    """Test the /statistics endpoint."""
    # Create a few comparisons
    client.post("/compare", json={"text": "What is the capital of France?"})
    client.post("/compare", json={"text": "What is the largest ocean?"})

    # Cast votes
    client.post("/vote/1/model_1")
    client.post("/vote/1/model_1")
    client.post("/vote/2/model_2")

    # Get statistics
    response = client.get("/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_comparisons"] == 2
    assert data["total_votes_model_1"] == 2
    assert data["total_votes_model_2"] == 1


def test_compare_invalid_payload(db_session):
    """Test /compare with invalid payload."""
    response = client.post("/compare", json={"invalid_key": "value"})
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
