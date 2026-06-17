import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pathlib import Path


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import os

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )
)

mlflow.set_experiment("adult-income")


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "adult.csv"

# ------------------------
# Load Data
#-------------------------

df = pd.read_csv(DATA_PATH)


# -----------------
# Clean Target 
#------------------


df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# -----------------
# Features 
# -----------------

X = df.drop('income', axis=1)
y = df['income']

# -----------------
# COlumn Types
# -----------------

categorical_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native-country'
]

numerical_features = [
    'age',
    'fnlwgt',
    'educational-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]


# -----------------
# Preprocessor
# -----------------


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# --------------------------------------------------
# Model
# --------------------------------------------------

N_ESTIMATORS = 200
MAX_DEPTH = 5
LEARNING_RATE = 0.05


model = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    random_state=42,
    eval_metric="logloss"
)


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

    
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# --------------------------------------------------
# Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Train
# --------------------------------------------------

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    
    mlflow.log_metric("accuracy", acc)
    
    signature = infer_signature(model_input=X_test, model_output=preds)
    
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model", # the directory path under mlflow run's artifact storage where model file will be saved
        signature=signature
    )
    
    
    
    
    client = MlflowClient()
    model_name = "IncomeModel"
    model_uri = model_info.model_uri
    print(f"Model URI: {model_uri}")
    
    try:
        client.create_registered_model(name=model_name)
    except Exception as e:
        print(e)
    
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version="1"
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    joblib.dump(
        pipeline,
        "artifacts/model.pkl"
    )

    
    mlflow.log_artifact(
        "artifacts/model.pkl"
    )
    
    print("Model Training Complete")



