import os
import sys
import glob
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python_operator import PythonOperator  # <-- 1.10.x import

# ----- Config -----
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow"))  # set in env if different
RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
PDF_SPLITTER_PATH = str(PROJECT_ROOT / "src" / "parse" / "pdf_splitter.py")
MIN_SIZE_MB = 3.0

def _already_split(pdf_path):
    """Check for idempotency: skip if first chunk or manifest exists."""
    p = Path(pdf_path)
    split_dir = p.parent / "split_pdfs"
    return (split_dir / f"{p.stem}_part_001.pdf").exists() or (split_dir / "split_manifest.json").exists()

def _infer_ticker(pdf_path):
    """
    Infer {TICKER} from path like /opt/airflow/data/raw/FINTBX/pdf/file.pdf
    Falls back to parent of 'pdf' directory if present.
    """
    parts = Path(pdf_path).parts
    if "raw" in parts:
        try:
            i = parts.index("raw")
            return parts[i + 1]
        except Exception:
            pass
    # Fallback: parent of pdf directory
    try:
        pdf_idx = parts.index("pdf")
        return parts[pdf_idx - 1]
    except Exception:
        return None

def find_large_pdfs(raw_dir=RAW_DIR, min_size_mb=MIN_SIZE_MB):
    """Find PDFs > min_size_mb and not already split."""
    pdfs = []
    for path in glob.glob(os.path.join(raw_dir, "**", "*.pdf"), recursive=True):
        try:
            if os.path.getsize(path) > min_size_mb * 1024 * 1024 and not _already_split(path):
                pdfs.append(path)
        except OSError:
            # race conditions or broken symlinks
            continue
    return pdfs

def split_large_pdfs_task(**context):
    """
    For each large PDF, call pdf splitter:
      pdf_splitter.py <pdf_file> --size 3 --ticker <TICKER> --output data/raw
    Uses sys.executable so we run in the worker's venv.
    """
    pdfs = find_large_pdfs()
    if not pdfs:
        print("No PDFs require splitting.")
        return

    for pdf_path in pdfs:
        ticker = _infer_ticker(pdf_path)
        cmd = [
            sys.executable, PDF_SPLITTER_PATH,
            pdf_path,
            "--size", str(MIN_SIZE_MB),
            "--output", RAW_DIR
        ]
        if ticker:
            cmd.extend(["--ticker", ticker])

        print("[split]", " ".join(cmd))
        # Prefer subprocess.run with check=True on 1.10.x as well
        import subprocess
        try:
            res = subprocess.run(cmd, check=True, text=True, capture_output=True)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
        except subprocess.CalledProcessError as e:
            print("[ERROR] Split failed for:", pdf_path)
            print(e.stdout or "")
            print(e.stderr or "")
            raise

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
    schedule_interval=None,   # <-- Airflow 1.10.x
    catchup=False,
) as dag:

    split_task = PythonOperator(
        task_id="split_large_pdfs_task",
        python_callable=split_large_pdfs_task,
        provide_context=True,  # <-- Airflow 1.10.x passes **kwargs this way
    )
