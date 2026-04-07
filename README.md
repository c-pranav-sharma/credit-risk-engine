# 🛡️ SentinelScore: End-to-End MLOps Credit Risk Engine

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?logo=mlflow&logoColor=blue)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_HPO-blueviolet)](https://optuna.org/)

## 📖 Overview
SentinelScore is a production-grade machine learning microservice designed to predict loan default risk. Moving beyond static Jupyter Notebooks, this project implements a complete **MLOps lifecycle**—from secure data engineering and pipeline versioning to Bayesian hyperparameter optimization and automated CI/CD deployment. 

The engine evaluates applicant financial data and external credit histories to calculate a highly optimized, cost-sensitive probability of default, dynamically maximizing bank profitability based on the F1-score.

## 🏗️ Architecture & Tech Stack
This system is built using enterprise-standard MLOps architecture:

* **Data Engineering:** `Pandas` (Multi-table joins, PII masking, feature aggregation)
* **Pipeline & Versioning:** `DVC` (Data Version Control)
* **Modeling & HPO:** `XGBoost`, `Optuna` (Tree-Structured Parzen Estimator)
* **Experiment Tracking:** `MLflow` (SQLite backend, Artifact Registry)
* **Production Serving:** `FastAPI`, `Uvicorn`
* **Containerization:** `Docker` (Minimal Linux Base)
* **Frontend UI:** `Streamlit`
* **CI/CD:** `GitHub Actions`

## ⚙️ The MLOps Pipeline
This project is orchestrated entirely via DVC, ensuring bit-for-bit reproducibility across all stages:

1. **Stage 1: Privacy Hardening (`prepare.py`)** Raw application data is ingested and PII (Customer IDs) is securely masked using SHA-256 HMAC hashing to ensure GDPR/CCPA compliance before any ML occurs.
2. **Stage 2: Feature Engineering (`featurize.py`)** Multiple relational databases (Bureau histories, past Installments) are aggregated and flattened into a high-signal feature matrix.
3. **Stage 3: Optimization & Registry (`train.py`)** Optuna autonomously hunts for the optimal XGBoost architecture. The winning model and an F1-maximized decision threshold are registered in MLflow, and physical artifacts are decoupled for deployment.
4. **Stage 4: Production Serving (`main.py`)** The physical model is injected into a high-speed FastAPI web server, guarded by robust schema validation.

## 📊 Production Performance
The current production candidate (`v3`) achieved the following metrics on a held-out test set:
* **ROC-AUC:** `0.7673`
* **PR-AUC:** `0.2603` (Highly imbalanced dataset)
* **Optimal Decision Threshold:** `0.6728`

## 🚀 Quickstart: Run Locally

### 1. The FastAPI Backend (Docker)
Build and spin up the isolated ML microservice:
```bash
# Build the container
docker build -t sentinelscore-api:v1 .

# Run the inference server
docker run -p 8000:8000 sentinelscore-api:v1
