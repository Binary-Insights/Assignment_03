# ===============================================
#   Dockerfile for Airflow + Docling OCR Parser
#   Multi-environment setup for dependency isolation
# ===============================================
# Base image: Airflow 2.10.4 with Python 3.11
FROM apache/airflow:2.10.4-python3.11

# ----------------------------
# System and environment setup
# ----------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/opt/airflow/workspace"

# Install system dependencies for PDF parsing, OCR, and OpenGL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    poppler-utils \
    ghostscript \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    virtualenv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Switch back to airflow user
# ----------------------------
USER airflow

# ----------------------------
# Airflow core + Amazon provider (main .venv)
# ----------------------------
ARG AIRFLOW_VERSION=2.10.4
ARG PYTHON_VERSION=3.11
ARG CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -c ${CONSTRAINTS_URL} \
    apache-airflow \
    apache-airflow-providers-amazon

# ----------------------------
# AWS SDKs and utilities
# ----------------------------
RUN pip install --no-cache-dir --prefer-binary \
    "boto3>=1.34,<2" \
    "awscli>=1.32,<2"

# ----------------------------
# Core scientific stack (safe ABI pins)
# ----------------------------
# Prevents "numpy.dtype size changed" errors.
RUN pip install --no-cache-dir --prefer-binary \
    "numpy==1.26.4" \
    "pandas==2.2.2"

# ----------------------------
# App-specific and parsing dependencies (main environment)
# ----------------------------
RUN pip install --no-cache-dir --prefer-binary \
    instructor \
    openai \
    "pydantic>=2.6,<3" \
    requests \
    beautifulsoup4 \
    python-dotenv \
    "dvc[s3]>=3.63.0" \
    "lxml>=4.9.3" \
    psutil \
    "pymupdf>=1.26.5" \
    "fastapi>=0.119.0" \
    "uvicorn[standard]>=0.27,<1" \
    "streamlit>=1.39.0" \
    "transformers>=4.34,<5" \
    wikipedia \
    langchain-openai \
    pinecone \
    psycopg2-binary \
    langsmith

# ----------------------------
# Optional: OCR / ML accelerators (main environment)
# ----------------------------
# Try GPU runtime first; fallback to CPU if unavailable
RUN pip install --no-cache-dir --prefer-binary "onnxruntime-gpu>=1.23.0" || echo "GPU runtime skipped"
RUN pip install --no-cache-dir --prefer-binary "onnxruntime>=1.17,<2"

# ----------------------------
# CREATE SEPARATE DOCLING VIRTUAL ENVIRONMENT
# ----------------------------
# This isolates docling dependencies to prevent conflicts
USER root
RUN mkdir -p /opt/venv_docling && \
    chown -R airflow:airflow /opt/venv_docling

USER airflow

# Create virtual environment for docling
RUN python -m venv /opt/venv_docling

# Install docling + its dependencies in isolated venv
# Use the specific constraints from requirements-docling.txt
RUN /opt/venv_docling/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv_docling/bin/pip install --no-cache-dir --prefer-binary \
    "docling==2.57.0" \
    "docling-core>=2.8.0" \
    "docling-parse>=2.4.0" \
    "pillow>=10.0.0" \
    "pdf2image>=1.16.0" \
    "rapidocr-onnxruntime>=1.3.0" \
    "easyocr>=1.6.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pydantic>=2.0.0" \
    "torch>=2.0.0" \
    "torchvision>=0.15.0"

# Create wrapper scripts for docling execution
RUN mkdir -p /opt/scripts && chown airflow:airflow /opt/scripts

USER root

# Create script to run docling with isolated venv
RUN cat > /opt/scripts/run_docling.sh << 'EOF'
#!/bin/bash
# Wrapper to run docling using isolated venv
set -e

# Activate docling venv
source /opt/venv_docling/bin/activate

# Run the docling extractor script
exec python "$@"
EOF

chmod +x /opt/scripts/run_docling.sh

# Create script for running Airflow tasks with docling
RUN cat > /opt/scripts/run_parsing_dag.sh << 'EOF'
#!/bin/bash
# Run parsing DAG with docling venv available
source /opt/venv_docling/bin/activate
export PYTHONPATH="/opt/airflow/workspace:$PYTHONPATH"
export PATH="/opt/venv_docling/bin:$PATH"
exec python "$@"
EOF

chmod +x /opt/scripts/run_parsing_dag.sh

# Create script for subprocess calls from Airflow
RUN cat > /opt/scripts/docling_parser.py << 'EOF'
#!/usr/bin/env python
"""
Wrapper to run docling_extractor_v2.py using subprocess
Used by parsing_dag.py to invoke docling in isolated environment
"""
import sys
import subprocess
import os

def run_docling_extractor(script_rel, ticker_folder, pdf_file):
    """Run docling extractor in isolated venv"""
    # Activate the docling venv and run the script
    cmd = [
        '/opt/venv_docling/bin/python',
        script_rel,
        ticker_folder,
        '--pdf',
        pdf_file,
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = '/opt/airflow/workspace'
    
    result = subprocess.run(cmd, env=env, cwd='/opt/airflow/workspace')
    sys.exit(result.returncode)

if __name__ == '__main__':
    run_docling_extractor(sys.argv[1], sys.argv[2], sys.argv[3])
EOF

chmod +x /opt/scripts/docling_parser.py

USER airflow

# ----------------------------
# Workspace structure
# ----------------------------
WORKDIR /opt/airflow/workspace
ENV PYTHONPATH="/opt/airflow/workspace"

# Add scripts to PATH
ENV PATH="/opt/scripts:$PATH"

# Copy all workspace files (DAGs, src/, data/, etc.)
COPY . /opt/airflow/workspace

