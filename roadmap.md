Tech Stack
Training
Python
Pandas
Scikit-learn
XGBoost
MLflow
Registry
MLflow Registry
Serving
FastAPI
Container
Docker
CI/CD
GitHub Actions
AWS
ECR
ECS Fargate
CloudWatch
IAM

No Kubernetes initially.

Kubernetes comes later.

Project Structure
project/

├── train/
│   ├── train.py
│   └── preprocess.py
│
├── serving/
│   ├── app.py
│   └── requirements.txt
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│
├── Dockerfile
│
├── mlflow/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
└── infrastructure/
Learning Roadmap
Phase 1

Build Model

Step 1

Train model locally

Output:

model.pkl
Step 2

Track experiments using MLflow

Output:

Run history
Metrics
Artifacts
Step 3

Register model

Output:

IncomeModel v1
Phase 2

Build Serving Layer

Step 4

FastAPI inference service

POST /predict
Step 5

Load model from registry

Not from local file.

Important distinction.

mlflow.pyfunc.load_model(...)
Phase 3

Containerization

Step 6

Dockerize service

docker build
Step 7

Health endpoint

/health
Step 8

Automated tests

pytest
Phase 4

AWS Deployment

Step 9

Create ECR repository

income-model
Step 10

Push Docker image

ECR
Step 11

Deploy ECS Fargate service

1 task
Step 12

Load Balancer

ALB
Phase 5

Continuous Deployment

Step 13

GitHub Actions Pipeline

Trigger:

Model promoted to Production

Pipeline:

Tests
 ↓
Build
 ↓
Push ECR
 ↓
Deploy ECS

Phase 6

Production Features

Step 14

CloudWatch Monitoring

Step 15

Model Version Monitoring

Step 16

Rollback

Model v2 bad

↓

Redeploy v1


Final Architecture
Data Scientist

      ↓

Train XGBoost

      ↓

MLflow Registry

      ↓

Promote Model

      ↓

GitHub Actions

      ↓

Docker Build

      ↓

AWS ECR

      ↓

AWS ECS

      ↓

FastAPI Service

      ↓

Users
Recommended learning sequence
Train model locally
MLflow experiment tracking
MLflow model registry
FastAPI serving that loads from registry
Docker
Automated tests
AWS ECR
AWS ECS Fargate
GitHub Actions CD pipeline
Monitoring and rollback

The key rule for this project: from Step 5 onward we never load model.pkl directly. Every deployment must obtain the model from the registry. This is the industry pattern you were asking about in the previous conversation when you noticed that loading a local model file bypasses the registry.