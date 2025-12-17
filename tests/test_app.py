import pytest
import numpy as np
from app.app import app


class DummyModel:
    def predict(self, data):
        return np.array([100000])


class DummyScaler:
    def transform(self, data):
        return data


@pytest.fixture
def client(monkeypatch):
    # Import inside fixture to avoid module-level import errors during collection
    from app.app import app
    
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
    # Data matching the new numerical form fields in app/app.py
    data = {
        'college_tier': '2',
        'city_score': '1',
        'role_manager': '0',
        'previous_ctc': '50000',
        'previous_job_change': '1',
        'graduation_marks': '75',
        'exp_months': '30'
    }
    rv = client.post("/predict", data=data)
    assert rv.status_code == 200
    assert b'Predicted Monthly Salary' in rv.data