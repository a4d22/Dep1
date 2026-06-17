import mlflow
import subprocess

subprocess.run([
    "mlflow",
    "server",
    "--backend-store-uri",
    "sqlite:///db/mlflow.db",
    "--default-artifact-root",
    "./artifacts",
    "--host",
    "0.0.0.0",
    "--port",
    "5000",
    "--allowed-hosts",
    "localhost:5000,127.0.0.1,host.docker.internal,host.docker.internal:5000"
])