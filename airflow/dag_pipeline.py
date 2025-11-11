from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "you",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG("smartcity_pipeline", start_date=datetime(2024,1,1), schedule_interval="@daily", default_args=default_args, catchup=False) as dag:
    etl = BashOperator(
        task_id="run_etl",
        bash_command="python /opt/project/etl/etl.py"
    )
    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/project/training/train.py"
    )
    eval_task = BashOperator(
        task_id="evaluate",
        bash_command="python /opt/project/training/evaluate.py"
    )
    deploy = BashOperator(
        task_id="deploy_service",
        bash_command="echo 'Deploy step: docker compose up -d' && exit 0"
    )

    etl >> train >> eval_task >> deploy