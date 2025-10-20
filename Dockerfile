# Use Airflow 2.10.4 with Python 3.11
FROM apache/airflow:2.10.4-python3.11

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=100

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git poppler-utils tesseract-ocr libmagic1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Airflow 2.10.4 constraints for Python 3.11
ARG AIRFLOW_VERSION=2.10.4
ARG PYTHON_VERSION=3.11
ARG CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# (Optional) keep for reference
COPY pyproject.toml /tmp/

# 1) Upgrade pip once
RUN python -m pip install --upgrade pip

# 2) Install Airflow-related extras UNDER CONSTRAINTS
RUN pip install --no-cache-dir --prefer-binary -c ${CONSTRAINTS_URL} \
    apache-airflow-providers-amazon

# 3) Install AWS SDKs with compatible pins (avoids botocore churn)
#    If you don't need the CLI inside the container, OMIT awscli entirely.
RUN pip install --no-cache-dir --prefer-binary \
    "boto3>=1.34,<2" \
    "awscli>=1.32,<2"

# 4) Install the rest (not tightly coupled to Airflow)
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
