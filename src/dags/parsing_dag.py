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
    dag_id="parsing_dag",
    description="Parse split PDFs sequentially per ticker; parallel across tickers.",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 10, 19),
    catchup=False,
    max_active_runs=1,
    tags=["pdf", "docling", "parse"],
) as dag:

    logger = logging.getLogger("parsing_dag")
    logger.setLevel(logging.INFO)

    @task(task_id="discover_ticker_parts")
    def discover_ticker_parts() -> List[Dict]:
        """
        Scan data/raw/*/pdf/split_pdfs for '*_part_*.pdf'
        Returns: [{ "ticker": "FINTBX", "parts": ["fintbx_part_001.pdf", ...] }, ...]
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
            if not parts:
                continue

            logger.info("Found %d parts for %s in %s", len(parts), ticker, split_dir)
            jobs.append({"ticker": ticker, "parts": parts})

        if not jobs:
            logger.warning("No split PDFs found. Check that your splitter populated %s/**/%s",
                           RAW_ROOT_REL, SPLIT_SUBPATH)
        return jobs

    @task(task_id="parse_ticker_sequential")
    def parse_ticker_sequential(job: Dict) -> str:
        """
        For one ticker, run the parser sequentially on each part:
          /opt/venv_docling/bin/python src/parse/docling_extractor_v2.py {RAW_ROOT_REL}/{TICKER}/pdf/split_pdfs --pdf <part>
        
        Uses isolated docling virtual environment to avoid dependency conflicts with instructor/openai
        """
        ticker: str = job["ticker"]
        parts: List[str] = job["parts"]

        base_dir = BASE_DIR
        script_rel = SCRIPT_REL
        relative_folder = Path(ticker) / SPLIT_SUBPATH

        logger.info("Starting parsing for %s: %d part(s)", ticker, len(parts))
        ok = 0
        for i, part in enumerate(parts, start=1):
            # Use isolated docling venv to avoid dependency conflicts
            cmd = [
                "/opt/venv_docling/bin/python",
                script_rel,
                str(relative_folder).replace("\\", "/"),
                "--pdf",
                part,
            ]
            logger.info("[%s] (%d/%d) Running: %s", ticker, i, len(parts), " ".join(cmd))

            # Setup environment with docling venv in PATH
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
                    logger.info("[%s] stdout: %s", ticker, res.stdout[:2000])
                if res.stderr:
                    logger.warning("[%s] stderr: %s", ticker, res.stderr[:2000])

                if res.returncode == 0:
                    ok += 1
                else:
                    logger.error("[%s] FAILED on part %s (exit=%s)", ticker, part, res.returncode)
            except Exception as e:
                logger.exception("[%s] Exception while parsing %s: %s", ticker, part, e)

        summary = f"{ticker}: {ok}/{len(parts)} parts parsed successfully"
        logger.info(summary)
        # Soft-fail behavior: we return summary even if some parts failed; adjust if you want hard fail
        return summary

    # Orchestration
    jobs = discover_ticker_parts()

    # Attach pool before expand (TaskFlow .override is the right way)
    parse_callable = parse_ticker_sequential
    if POOL_NAME:
        parse_callable = parse_ticker_sequential.override(pool=POOL_NAME)

    ticker_summaries = parse_callable.expand(job=jobs)
