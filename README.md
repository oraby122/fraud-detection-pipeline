# Fraud Detection Pipeline

An end-to-end fraud detection system built with:
- **Apache Airflow** for workflow orchestration
- **Python / scikit-learn / XGBoost** for model training
- **Docker** for reproducibility and deployment

## 📂 Project Structure
'''
.
├── README.md
├── data
│   └── data.tar.gz
├── docker
│   ├── airflow
│   │   ├── Dockerfile
│   │   ├── dags
│   │   ├── docker-compose.yaml
│   │   └── logs
│   └── app
│       ├── Dockerfile
│       └── requirements.txt
└── src
    ├── data_prep
    │   ├── data_inspection.py
    │   └── feature_engineering.py
    ├── models
    │   └── model.py
    └── utils
'''
