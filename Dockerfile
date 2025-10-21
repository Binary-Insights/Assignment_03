# ===============================================
#   Dockerfile for Airflow + Docling OCR Parser
# ===============================================
# Base image: Airflow 2.10.4 with Python 3.11
FROM apache/airflow:2.10.4-python3.11

# ----------------------------
# System and environment setup
# ----------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

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
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Switch back to airflow user
# ----------------------------
USER airflow

# ----------------------------
# Airflow core + Amazon provider
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
# App-specific and parsing dependencies
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
    docling==2.57.0 \
    psutil \
    "pymupdf>=1.26.5" \
    "fastapi>=0.119.0" \
    "uvicorn[standard]>=0.27,<1" \
    "streamlit>=1.39.0" \
    "transformers>=4.34,<5"

# ----------------------------
# Optional: OCR / ML accelerators
# ----------------------------
# Try GPU runtime first; fallback to CPU if unavailable
RUN pip install --no-cache-dir --prefer-binary "onnxruntime-gpu>=1.23.0" || echo "GPU runtime skipped"
RUN pip install --no-cache-dir --prefer-binary "onnxruntime>=1.17,<2"

# OCR engines (RapidOCR + EasyOCR)
RUN pip install --no-cache-dir --prefer-binary "rapidocr-onnxruntime>=1.3.0" easyocr

# ----------------------------
# Workspace structure
# ----------------------------
WORKDIR /opt/airflow/workspace
ENV PYTHONPATH="/opt/airflow/workspace"


# Copy all workspace files (DAGs, src/, data/, etc.)
COPY . /opt/airflow/workspace

