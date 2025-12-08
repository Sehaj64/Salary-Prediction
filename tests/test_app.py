import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    """Test the home page."""
    rv = client.get("/")
    assert rv.status_code == 200


def test_predict(client):
    """Test the predict endpoint."""
    data = {
        "College": "3",
        "City": "1",
        "Previous CTC": "55000",
        "Previous job change": "2",
        "Graduation Marks": "75",
        "EXP (Month)": "24",
        "Role": "1",
    }
    rv = client.post("/predict", data=data)
    assert rv.status_code == 200
    assert b"Predicted Salary (CTC)" in rv.data

