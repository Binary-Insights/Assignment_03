import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task

# ----------------------------
# Configuration (old-style, static)
# ----------------------------
BASE_DIR = Path("/opt/airflow/workspace")
RAW_ROOT_REL = "data/raw"
SPLIT_SUBPATH = "pdf/split_pdfs"
PART_GLOB = "*_part_*.pdf"
SCRIPT_REL = "src/parse/docling_extractor_v2.py"

# Optional pool (create in UI: Admin -> Pools)
POOL_NAME = "parsing_pool"   # set to None to use default pool

# Where we expect parsed results to land; we pre-create per-ticker dirs
PARSED_ROOT_REL = "data/parsed"

default_args = {
    "owner": "data_engineer_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="parsing_dag_v2",
    description="Parse split PDFs in parallel, one Airflow task per PDF part.",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 10, 19),
    catchup=False,
    max_active_runs=1,
    tags=["pdf", "docling", "parse"],
) as dag:

    logger = logging.getLogger("parsing_dag_v2")
    logger.setLevel(logging.INFO)

    @task(task_id="discover_parts_parallel")
    def discover_parts_parallel() -> List[Dict]:
        """
        Scan data/raw/*/pdf/split_pdfs for '*_part_*.pdf'
        Returns: [{ "ticker": "FINTBX", "part": "fintbx_part_001.pdf" }, ...]
        """
        raw_root = BASE_DIR / RAW_ROOT_REL
        if not raw_root.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_root}")

        jobs: List[Dict] = []
        for ticker_dir in sorted(raw_root.glob("*")):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            split_dir = ticker_dir / SPLIT_SUBPATH
            if not split_dir.exists():
                continue
            parts = sorted(p.name for p in split_dir.glob(PART_GLOB) if p.is_file())
            for part in parts:
                jobs.append({"ticker": ticker, "part": part})
        if not jobs:
            logger.warning("No split PDFs found. Check that your splitter populated %s/**/%s", RAW_ROOT_REL, SPLIT_SUBPATH)
        return jobs

    @task(task_id="parse_part_parallel")
    def parse_part_parallel(job: Dict) -> str:
        """
        Parse a single PDF part using docling in isolated venv.
        """
        ticker: str = job["ticker"]
        part: str = job["part"]
        base_dir = BASE_DIR
        script_rel = SCRIPT_REL
        relative_folder = Path(ticker) / SPLIT_SUBPATH
        cmd = [
            "/opt/venv_docling/bin/python",
            script_rel,
            str(relative_folder).replace("\\", "/"),
            "--pdf",
            part,
        ]
        logger.info(f"[PARSE] {ticker} - {part}: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PARSED_ROOT"] = str(BASE_DIR / PARSED_ROOT_REL)
        env["PYTHONPATH"] = f"{str(BASE_DIR)}:/opt/venv_docling/lib/python3.11/site-packages"
        env["PATH"] = f"/opt/venv_docling/bin:{env.get('PATH', '')}"
        try:
            res = subprocess.run(
                cmd,
                cwd=str(base_dir),
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
            if res.stdout:
                logger.info(f"[{ticker}] stdout: {res.stdout[:2000]}")
            if res.stderr:
                logger.warning(f"[{ticker}] stderr: {res.stderr[:2000]}")
            if res.returncode == 0:
                return f"{ticker}: {part} parsed successfully"
            else:
                logger.error(f"[{ticker}] FAILED on part {part} (exit={res.returncode})")
                return f"{ticker}: {part} failed"
        except Exception as e:
            logger.exception(f"[{ticker}] Exception while parsing {part}: {e}")
            return f"{ticker}: {part} exception"

    jobs = discover_parts_parallel()
    parse_callable = parse_part_parallel
    if POOL_NAME:
        parse_callable = parse_part_parallel.override(pool=POOL_NAME)
    part_summaries = parse_callable.expand(job=jobs)
