"""
Master Orchestration DAG
Sequentially runs: split_large_pdfs → parsing_dag → storing_dag
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
import logging

default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'master_orchestration_dag',
    default_args=default_args,
    description='Master DAG: Sequentially run split_large_pdfs → parsing_dag → storing_dag',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['orchestration', 'master'],
)

logger = logging.getLogger(__name__)

# ============================================================================
# Task 1: Trigger split_large_pdfs DAG
# ============================================================================
trigger_split_pdfs = TriggerDagRunOperator(
    task_id='trigger_split_large_pdfs',
    trigger_dag_id='split_large_pdfs',
    dag=dag,
    wait_for_completion=True,  # Wait for the triggered DAG to complete
    poke_interval=30,  # Check status every 30 seconds
)

# ============================================================================
# Task 2: Wait for split_large_pdfs to complete, then trigger parsing_dag
# ============================================================================
trigger_parsing = TriggerDagRunOperator(
    task_id='trigger_parsing_dag',
    trigger_dag_id='parsing_dag',
    dag=dag,
    wait_for_completion=True,  # Wait for the triggered DAG to complete
    poke_interval=30,  # Check status every 30 seconds
)

# ============================================================================
# Task 3: Wait for parsing_dag to complete, then trigger storing_dag
# ============================================================================
trigger_storing = TriggerDagRunOperator(
    task_id='trigger_storing_dag',
    trigger_dag_id='storing_dag',
    dag=dag,
    wait_for_completion=True,  # Wait for the triggered DAG to complete
    poke_interval=30,  # Check status every 30 seconds
)

# ============================================================================
# Set task dependencies (sequential execution)
# ============================================================================
trigger_split_pdfs >> trigger_parsing >> trigger_storing

# ============================================================================
# Task summary logging
# ============================================================================
if __name__ == "__main__":
    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         MASTER ORCHESTRATION DAG                               ║
    ║                                                                ║
    ║  Execution Flow (Sequential):                                  ║
    ║                                                                ║
    ║  1️⃣  trigger_split_large_pdfs                                  ║
    ║      └─→ Splits large PDF files                                ║
    ║          Output: data/raw/* (split PDFs)                       ║
    ║                                                                ║
    ║  2️⃣  trigger_parsing_dag                                       ║ 
    ║          Input: data/raw/*                                     ║
    ║          Output: data/parsed/*                                 ║
    ║                                                                ║
    ║  3️⃣  trigger_storing_dag                                      ║
    ║      └─→ Uploads parsed files to S3                            ║
    ║          Input: data/parsed/*                                  ║
    ║          S3: s3://bucket/assignment_03/parsed/*                ║
    ║                                                                ║
    ║  Status: Each task waits for previous to complete              ║ 
    ║  Trigger: Manual (no schedule)                                 ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
