FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY serving ./serving

COPY artifacts/model.pkl ./artifacts/model.pkl

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]