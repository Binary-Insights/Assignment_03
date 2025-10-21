import os
import sys
import glob
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator

# ----- Configuration -----
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow"))
RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
PDF_SPLITTER_PATH = str(PROJECT_ROOT / "src" / "parse" / "pdf_splitter.py")
PAGES_PER_SPLIT = 100  # default split every 100 pages

def _already_split(pdf_path: str) -> bool:
    """Skip if already split (idempotency)."""
    p = Path(pdf_path)
    split_dir = p.parent / "split_pdfs"
    return (split_dir / f"{p.stem}_part_001.pdf").exists() or (split_dir / "split_manifest.json").exists()

def _infer_ticker(pdf_path: str):
    """Infer ticker symbol from path like data/raw/<TICKER>/pdf/file.pdf."""
    parts = Path(pdf_path).parts
    if "raw" in parts:
        try:
            i = parts.index("raw")
            return parts[i + 1]
        except Exception:
            pass
    try:
        pdf_idx = parts.index("pdf")
        return parts[pdf_idx - 1]
    except Exception:
        return None

def find_pdfs(raw_dir=RAW_DIR):
    """Find all PDFs not already split."""
    pdfs = []
    for path in glob.glob(os.path.join(raw_dir, "**", "*.pdf"), recursive=True):
        try:
            if not _already_split(path):
                pdfs.append(path)
        except OSError:
            continue
    return pdfs

def split_large_pdfs_task(**context):
    """Split all PDFs by fixed number of pages using --pages argument."""
    pdfs = find_pdfs()
    if not pdfs:
        print("No PDFs found for splitting.")
        return

    for pdf_path in pdfs:
        ticker = _infer_ticker(pdf_path)
        cmd = [
            sys.executable, PDF_SPLITTER_PATH,
            pdf_path,
            "--pages", str(PAGES_PER_SPLIT),
            "--output", RAW_DIR
        ]
        if ticker:
            cmd.extend(["--ticker", ticker])

        print("[split]", " ".join(cmd))
        try:
            res = subprocess.run(cmd, check=True, text=True, capture_output=True)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Split failed for {pdf_path}")
            print(e.stdout or "")
            print(e.stderr or "")
            raise

# ----- Airflow DAG -----
default_args = {
    "owner": "data_engineer_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="split_large_pdfs",
    default_args=default_args,
    start_date=datetime(2025, 10, 19),
    schedule_interval=None,
    catchup=False,
) as dag:

    split_task = PythonOperator(
        task_id="split_large_pdfs_task",
        python_callable=split_large_pdfs_task,
        provide_context=True,
    )
