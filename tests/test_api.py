from fastapi.testclient import TestClient
from serving.app import app


client = TestClient(app)
def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"
    

# def test_predict():

#     payload = {
#         "age": 39,
#         "workclass": "Private",
#         "fnlwgt": 77516,
#         "education": "Bachelors",
#         "educational_num": 13,
#         "marital_status": "Never-married",
#         "occupation": "Adm-clerical",
#         "relationship": "Not-in-family",
#         "race": "White",
#         "gender": "Male",
#         "capital_gain": 2174,
#         "capital_loss": 0,
#         "hours_per_week": 40,
#         "native_country": "United-States"
#     }

#     response = client.post(
#         "/predict",
#         json=payload
#     )

#     assert response.status_code == 200

#     prediction = response.json()["prediction"]

#     assert prediction in [0, 1]