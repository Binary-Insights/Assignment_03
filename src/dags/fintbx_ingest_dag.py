"""
Airflow DAG: Run experimental_framework.py and ingest_to_pinecone.py
End-to-end RAG pipeline for chunk generation + Pinecone ingestion.
"""

import os
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.decorators import task

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = Path("/opt/airflow/workspace")
SRC_DIR = BASE_DIR / "src" / "rag"
PYTHON_EXE = "python"

TICKER = os.getenv("TICKER", "FINTBX")  # default ticker

default_args = {
    "owner": "data_engineer_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

logger = logging.getLogger("rag_ingestion_dag")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# DAG Definition
# -------------------------------------------------------------------
with DAG(
    dag_id="rag_ingestion_dag",
    description="Runs experimental_framework.py + ingest_to_pinecone.py sequentially for full RAG ingestion",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 10, 21),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "pinecone", "docling", "embedding"],
) as dag:

    # -----------------------------
    # 1️⃣ Run experimental framework
    # -----------------------------
    @task(task_id="run_experimental_framework")
    def run_experimental_framework():
        cmd = [
            PYTHON_EXE,
            str(SRC_DIR / "experimental_framework.py"),
            TICKER
        ]
        logger.info(f"🚀 Running experimental framework: {' '.join(cmd)}")

        res = subprocess.run(cmd, cwd=str(BASE_DIR), text=True, capture_output=True)
        if res.returncode != 0:
            logger.error(f"❌ experimental_framework failed:\n{res.stderr}")
            raise RuntimeError(f"Experimental Framework failed: {res.stderr}")
        logger.info(f"✅ experimental_framework completed:\n{res.stdout[:2000]}")

    # -----------------------------
    # 2️⃣ Run Pinecone ingestion
    # -----------------------------
    @task(task_id="run_ingest_to_pinecone")
    def run_ingest_to_pinecone():
        cmd = [
            PYTHON_EXE,
            str(SRC_DIR / "ingest_to_pinecone.py")
        ]
        logger.info(f"🚀 Running Pinecone ingestion: {' '.join(cmd)}")

        res = subprocess.run(cmd, cwd=str(BASE_DIR), text=True, capture_output=True)
        if res.returncode != 0:
            logger.error(f"❌ ingest_to_pinecone failed:\n{res.stderr}")
            raise RuntimeError(f"Ingest to Pinecone failed: {res.stderr}")
        logger.info(f"✅ ingest_to_pinecone completed:\n{res.stdout[:2000]}")

    # -----------------------------
    # DAG Orchestration
    # -----------------------------
    exp_task = run_experimental_framework()
    ingest_task = run_ingest_to_pinecone()

    exp_task >> ingest_task  # sequential dependency

