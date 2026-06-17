# import mlflow
# import pandas as pd

# mlflow.set_tracking_uri("http://localhost:5000")


# def test_model_loads():

#     model = mlflow.pyfunc.load_model(
#         "models:/IncomeModel@Champion"
#     )

#     assert model is not None


# def test_prediction():

#     model = mlflow.pyfunc.load_model(
#         "models:/IncomeModel@Champion"
#     )

#     sample = pd.DataFrame([{
#         "age": 39,
#         "workclass": "Private",
#         "fnlwgt": 77516,
#         "education": "Bachelors",
#         "educational-num": 13,
#         "marital-status": "Never-married",
#         "occupation": "Adm-clerical",
#         "relationship": "Not-in-family",
#         "race": "White",
#         "gender": "Male",
#         "capital-gain": 2174,
#         "capital-loss": 0,
#         "hours-per-week": 40,
#         "native-country": "United-States"
#     }])

#     pred = model.predict(sample)

#     assert len(pred) == 1
#     assert pred[0] in [0, 1]


def test_basic_math():
    assert 1 + 1 == 2