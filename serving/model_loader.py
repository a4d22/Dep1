# import os
# import mlflow

# mlflow.set_tracking_uri(
#     os.getenv(
#         "MLFLOW_TRACKING_URI",
#         "http://localhost:5000"
#     )
# )

# _model = None


# def get_model():

#     global _model

#     if _model is None:

#         _model = mlflow.pyfunc.load_model(
#             "models:/IncomeModel@Champion"
#         )

#     return _model

import joblib

_model = None

def get_model():

    global _model

    if _model is None:
        _model = joblib.load(
            "artifacts/model.pkl"
        )

    return _model