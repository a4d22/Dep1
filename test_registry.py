import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.pyfunc.load_model(
    model_uri="models:/IncomeModel@Champion"
)

print(type(model))


from mlflow import MlflowClient

client = MlflowClient("http://localhost:5000")

for mv in client.search_model_versions("name='IncomeModel'"):
    print("Version:", mv.version)
    print("Source:", mv.source)