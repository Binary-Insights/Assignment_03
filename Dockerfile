# ===============================================
#   Dockerfile for Airflow + Docling OCR Parser
#   Multi-environment setup for dependency isolation
# ===============================================
FROM apache/airflow:2.10.4-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/opt/airflow/workspace"

# ---- System dependencies for Docling, OCR, and visualization ----
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

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ----------------------------
# Switch back to airflow user
# ----------------------------
USER airflow
ENV PATH="/root/.cargo/bin:$PATH"

# ----------------------------
# Airflow core + Amazon provider (main .venv)
# ----------------------------
ARG AIRFLOW_VERSION=2.10.4
ARG PYTHON_VERSION=3.11
ARG CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

RUN python -m pip install --upgrade pip setuptools wheel \
 && uv pip install -c ${CONSTRAINTS_URL} \
    apache-airflow \
    apache-airflow-providers-amazon

# ----------------------------
# AWS SDKs and utilities
# ----------------------------
RUN uv pip install \
    "boto3>=1.34,<2" \
    "awscli>=1.32,<2"

# ----------------------------
# Core scientific stack (safe ABI pins)
# ----------------------------
# Prevents "numpy.dtype size changed" errors.
RUN uv pip install \
    "numpy==1.26.4" \
    "pandas==2.2.2"

# ----------------------------
# App-specific and parsing dependencies (main environment)
# ----------------------------
RUN uv pip install \
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
    langchain \
    wikipedia \
    langchain-openai \
    langchain-text-splitters \
    pinecone \
    psycopg2-binary \
    langsmith

# ----------------------------
# Optional: OCR / ML accelerators (main environment)
# ----------------------------
# Try GPU runtime first; fallback to CPU if unavailable
RUN uv pip install "onnxruntime-gpu>=1.23.0" || echo "GPU runtime skipped"
RUN uv pip install "onnxruntime>=1.17,<2"

# ----------------------------
# CREATE SEPARATE DOCLING VIRTUAL ENVIRONMENT
# ----------------------------
# This isolates docling dependencies to prevent conflicts
USER root
RUN mkdir -p /opt/venv_docling && \
    python -m venv /opt/venv_docling && \
    chown -R airflow /opt/venv_docling

USER airflow

# Install docling + its dependencies in isolated venv
RUN /opt/venv_docling/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv_docling/bin/pip install uv && \
    /opt/venv_docling/bin/uv pip install \
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
USER root

RUN mkdir -p /opt/scripts && chown airflow /opt/scripts

# Create script to run docling with isolated venv
RUN echo '#!/bin/bash' > /opt/scripts/run_docling.sh && \
    echo 'set -e' >> /opt/scripts/run_docling.sh && \
    echo 'source /opt/venv_docling/bin/activate' >> /opt/scripts/run_docling.sh && \
    echo 'exec python "$@"' >> /opt/scripts/run_docling.sh && \
    chmod +x /opt/scripts/run_docling.sh

# Create wrapper script for Airflow subprocess calls
RUN echo '#!/bin/bash' > /opt/scripts/run_parsing_with_docling.sh && \
    echo 'source /opt/venv_docling/bin/activate' >> /opt/scripts/run_parsing_with_docling.sh && \
    echo 'export PYTHONPATH="/opt/airflow/workspace:$PYTHONPATH"' >> /opt/scripts/run_parsing_with_docling.sh && \
    echo 'export PATH="/opt/venv_docling/bin:$PATH"' >> /opt/scripts/run_parsing_with_docling.sh && \
    echo 'exec python "$@"' >> /opt/scripts/run_parsing_with_docling.sh && \
    chmod +x /opt/scripts/run_parsing_with_docling.sh

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