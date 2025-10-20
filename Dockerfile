# Use Airflow 2.10.4 with Python 3.11 (supports Pydantic 2.x)
FROM apache/airflow:2.10.4-python3.11

# Switch to root to install system dependencies if needed
USER root

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    gosu \
    wget \
    gnupg \
    unzip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Create cache directories with proper permissions
RUN mkdir -p /home/airflow/.cache/huggingface && \
    mkdir -p /home/airflow/.cache/deepsearch_glm

# Copy pyproject.toml (single source of truth)
COPY pyproject.toml /tmp/

# Install dependencies from pyproject.toml
# Note: We pin apache-airflow==2.10.4 to match the base image version
# This prevents pip from uninstalling the existing airflow during dependency resolution
RUN cd /tmp && \
    pip install --no-cache-dir \
    apache-airflow==2.10.4 \
    instructor \
    apache-airflow-providers-amazon \
    boto3 \
    openai \
    pydantic \
    requests \
    beautifulsoup4 \
    python-dotenv \
    "dvc[s3]" \
    lxml \
    docling==1.20.0 \
    psutil

# Pre-download the deepsearch_glm model to avoid runtime download issues
# RUN echo "try:" > /tmp/preload.py && \
#     echo "    from docling.document_converter import DocumentConverter" >> /tmp/preload.py && \
#     echo "    converter = DocumentConverter()" >> /tmp/preload.py && \
#     echo "    print('Models pre-downloaded successfully')" >> /tmp/preload.py && \
#     echo "except Exception as e:" >> /tmp/preload.py && \
#     echo "    print('Model pre-download failed:', str(e))" >> /tmp/preload.py && \
#     python /tmp/preload.py && \
#     rm /tmp/preload.py
