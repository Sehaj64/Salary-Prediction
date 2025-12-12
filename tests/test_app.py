import pytest
from app.app import app
import numpy as np


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
    data = {
        'college': '3',
        'city': '1',
        'previous_ctc': '55000',
        'previous_job_change': '2',
        'graduation_marks': '75',
        'exp_months': '24',
        'role': '1'
    }
    rv = client.post("/predict", data=data)
    assert rv.status_code == 200
    assert b'Predicted Monthly Salary' in rv.data