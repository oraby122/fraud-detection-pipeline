import os
import logging
import pandas as pd

from glob import glob
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine

DATA_FOLDER = "/opt/airflow/data/data_raw"
POSTGRES_CONN = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
TABLE_NAME = "fraud_detection.transactions"

data_files = glob("/opt/airflow/data/data_raw/*.pkl")
if not data_files:
    raise ValueError(f"No pickle files found in {DATA_FOLDER}")
else:
    logging.info(f"Found {len(data_files)} pkl files")

def aggregate_and_store():
    logging.info(f"Reading pickle files from {DATA_FOLDER}")
    # Read all .pkl files
    all_data = []
    for file in data_files:
        logging.info(f"Reading {file}")
        df = pd.read_pickle(os.path.join(DATA_FOLDER, file))
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # Clean column names
    full_df.columns = full_df.columns.str.lower().str.replace(" ", "_")
    logging.info(f"Total rows to insert: {len(full_df)}")
    # Push to Postgres
    engine = create_engine(POSTGRES_CONN)
    with engine.connect() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS fraud_detection;")
    full_df.to_sql("transactions", 
                    con=engine,
                    schema="fraud_detection", 
                    if_exists="replace",
                    index=False,
                    method='multi',
                    chunksize=10000)
    logging.info("Data inserted successfully.")


default_args = {
    'start_date': datetime(2023, 1, 1),
    'catchup': False,
}

with DAG(dag_id="fraud_detection_pipeline",
        default_args=default_args,
        schedule_interval=None,
        description="Aggregate pickle files and load into Postgres",) as dag:
    ingest_data = PythonOperator(
        task_id="aggregate_and_store_pickle_data",
        python_callable=aggregate_and_store,
    )
    ingest_data
