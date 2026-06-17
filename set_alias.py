# set_alias.py

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name="IncomeModel",
    alias="Champion",
    version="1"
)

print("Alias assigned.")