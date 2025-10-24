import os
import logging
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow?options=-csearch_path%3Dfraud_detection"
)
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def feature_engineering():
    # Connect to PostgreSQL using Airflow connection
    # hook = PostgresHook(postgres_conn_id='postgres_default')
    # conn = hook.get_conn()

    # Query the transaction table
    df = pd.read_sql("SELECT * FROM transactions", engine)

    logging.info(df.head())
    # Convert TX_DATETIME to datetime if not already
    df['tx_datetime'] = pd.to_datetime(df['tx_datetime'])

    # IS_WEEKEND_TRANSACTION
    df['IS_WEEKEND_TRANSACTION'] = df['tx_datetime'].dt.weekday >= 5

    # IS_NIGHT_TRANSACTION
    df['IS_NIGHT_TRANSACTION'] = df['tx_datetime'].dt.hour.between(0, 6)

    # Sort for rolling operations
    df = df.sort_values(['customer_id', 'tx_datetime'])

    # Rolling window features
    df.set_index('tx_datetime', inplace=True)

    def add_rolling_features(group):
        for days in [1, 7, 30]:
            rolling = group.rolling(f'{days}D')
            group[f'NUM_TRANSACTIONS_LAST_{days}DAY'] = rolling['transaction_id'].count().values
            group[f'AVG_AMOUNT_LAST_{days}DAY'] = rolling['tx_amount'].mean().values
        return group

    df = df.groupby('customer_id').apply(add_rolling_features)

    # Reset index for output
    df.reset_index(drop=True,inplace=True)

    # Drop if any NaNs in engineered features
    df = df.dropna()

    # Write to a new table
    df.to_sql("transactions_features", engine, if_exists="replace", index=False,chunksize = 10000)

def _print_done():
    print("Feature engineering completed and data stored in 'transactions_features'.")

with DAG(
    dag_id='feature_engineering_dag',
    default_args=default_args,
    description='Feature engineering DAG for fraud detection dataset',
    schedule_interval=None,  # Run manually or trigger externally
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['fraud', 'feature_engineering'],
) as dag:

    fe_task = PythonOperator(
        task_id='run_feature_engineering',
        python_callable=feature_engineering
    )

    done_task = PythonOperator(
        task_id='print_done',
        python_callable=_print_done
    )

    fe_task >> done_task
