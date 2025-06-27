# 💳 Real-Time Fraud Detection Pipeline

An end-to-end MLOps project for detecting credit card fraud in real time using modern ML engineering best practices.

---

## Project Overview

This project simulates a real-time fraud detection pipeline using AWS services, MLOps tooling, and robust monitoring. It includes:

- ⚙️ **DVC** for data versioning  
- 🧪 **MLflow** & 🪄 **Weights & Biases** for experiment tracking  
- 🧠 **ONNX** for portable model inference  
- 🧬 **Hydra** for configuration management  
- ⚓ **GitHub Actions** for CI/CD  
- 🐍 **uv** for Python environment management  
- 📊 **Evidently AI** for data drift monitoring  
- 🛠️ **Lambda + SQS** for serverless real-time inference

---

## Tools & Libraries Used

| Category           | Tool/Library                               |
|--------------------|---------------------------------------------|
| Data Versioning    | `DVC`, `AWS S3`                             |
| Model Registry     | `MLflow`, `ONNX`                            |
| Experiment Tracking| `Weights & Biases`, `Hydra`, `Matplotlib`  |
| Model Training     | `scikit-learn`, `PyTorch Lightning`, `SMOTE`|
| Data Monitoring    | `Evidently AI`, `CloudWatch`     |
| Inference          | `AWS Lambda`, `ONNX Runtime`, `SQS`        |
| CI/CD              | `GitHub Actions`                           |
| Environment Mgmt   | `uv`, `pipx`                                |

---


## ⚙️ Setup & Installation

### Python Environment with `uv`

```bash
pipx install uv
uv venv .venv
uv pip install pytorch-lightning torch pandas scikit-learn numpy hydra-core matplotlib seaborn dvc wandb onnx onnxruntime kaggle
uv add pytorch-lightning torch pandas scikit-learn numpy hydra-core matplotlib seaborn dvc wandb onnx onnxruntime kaggle
```
### Download Dataset from Kaggle
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```
### Data Preprocessing
```bash
uv run src/preprocessing.py
```
### Data Versioning with DVC
```bash
dvc init

dvc remote add -d s3remote s3://<your-bucket-name>/dvcstore
dvc remote modify s3remote endpointurl https://s3.us-west-1.amazonaws.com
dvc remote modify s3remote region us-west-1

dvc add data/raw/creditcard.csv
dvc add data/processed/X_train.csv
dvc add data/processed/X_test.csv
dvc add data/processed/Y_train.csv
dvc add data/processed/Y_test.csv

git add .gitignore *.dvc dvc.yaml dvc.lock
git commit -m "Added data files tracked with DVC"
dvc push
```
### Training
#### Default Training (no sweep)
```bash
uv python src/train.py
```
You can also use docker to run the training scripts,
```bash
docker build -t fraud-detection-app:latest .
docker run fraud-detection-app:latest
```
####  With Weights & Biases Sweep
```bash
wandb sweep sweep.yaml
wandb agent <sweep-agent-endpoint>
```
### Track Experiments
#### MLflow UI
```bash
mlflow ui
```
#### Weights & Biases Dashboard
Includes training loss, AUC, hyperparameter optimization, model comparison, etc.

### Real-Time Data Simulation
- Simulates credit card transactions
- Pushes JSON payloads to SQS queue
- Triggers AWS Lambda for real-time fraud prediction
```bash
python realtime_data_simulation.py
```
### Monitoring & Alerting
- Lambda logs → CloudWatch
- Open results/drift_report.html in web browser to monitor datadrift in real time data.

---
# 👤 Author
Raahul Krishna Durairaju
Machine Learning & MLOps Practitioner | MS CS @ Cal State Fullerton

🔗 [LinkedIn](https://www.linkedin.com/in/raahulkrishna/) • [GitHub](https://github.com/RAAHUL-tech)
