# ===============================================
# Airflow + Docling + Pinecone (managed by uv)
# ===============================================
FROM apache/airflow:2.10.4-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive

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
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Install uv package manager ----
RUN pip install --no-cache-dir uv

# Switch to airflow user (Airflow images run as non-root)
USER airflow

# ---- Workspace setup ----
WORKDIR /opt/airflow/workspace
ENV PYTHONPATH="/opt/airflow/workspace"

# ---- Copy dependency file first for caching ----
COPY pyproject.toml ./

# ---- Install dependencies (RAG + pipeline deps) using uv ----
RUN uv pip install --system \
      "langchain-core>=0.3.0,<1.0" \
      "langchain-text-splitters>=0.3.0,<1.0" \
      "langchain-openai>=0.2.0,<1.0" \
      "openai>=1.37,<2.0" \
      "pinecone-client>=5.0,<6.0" \
      "fastapi>=0.110,<1.0" \
      "uvicorn[standard]>=0.27,<1.0" \
      "requests>=2.31,<3.0" \
      "httpx>=0.27,<1.0" \
      "aiohttp>=3.9,<4.0" \
      "python-dotenv>=1.0,<2.0" \
      "pydantic>=2.6,<3.0" \
      "pydantic-settings>=2.2,<3.0" \
      "streamlit>=1.32,<2.0" \
      "tqdm>=4.66,<5.0"

# ---- Copy project source code ----
COPY . /opt/airflow/workspace