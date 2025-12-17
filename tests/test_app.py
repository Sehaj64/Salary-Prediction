import pytest
import sys
import os
import numpy as np

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import app  # noqa: E402


class DummyModel:
    def predict(self, data):
        return np.array([100000])


class DummyScaler:
    def transform(self, data):
        return data


@pytest.fixture
def client(monkeypatch):
    def dummy_get_model():
        return DummyModel()

    def dummy_get_scaler():
        return DummyScaler()

    monkeypatch.setattr('app.app.get_model', dummy_get_model)
    monkeypatch.setattr('app.app.get_scaler', dummy_get_scaler)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    rv = client.get("/")
    assert rv.status_code == 200


def test_predict(client):
    # Data matching the form fields in app/app.py
    data = {
        'education_level': "Bachelor's",
        'job_title': 'Software Engineer',
        'years_of_experience': '2.5',
        'previous_ctc': '50000',
        'previous_job_change': '1',
        'graduation_marks': '75'
    }
    rv = client.post("/predict", data=data)
    assert rv.status_code == 200
    assert b'Predicted Monthly Salary' in rv.data
