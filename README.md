# 🧠 Project AURELIA: Financial Concept Extraction Pipeline  
**DAMG 7245 – Fall 2025 – Binary Insights**

## Overview
Automates the parsing, chunking, embedding, retrieval, and structured-note generation of financial concepts from the *Financial Toolbox User’s Guide (fintbx.pdf)*.  
Powered by **Apache Airflow**, **OpenAI GPT-4o**, **Docling**, **LangChain**, **Pinecone**, and **AWS S3**, all containerized and orchestrated on **AWS EC2 (Docker Host)** (Assignment 3 – Project AURELIA, n.d.).

---

---

## Setup Instructions

### 1. Prerequisites
- **RAM ≥ 8 GB**
- **Docker Desktop** (Win/Mac) or Docker Engine + Compose (Linux)
- **Git**
- **AWS Account** (for S3 storage & IAM role)
- **OpenAI API Key**

---

### 2. Clone the Repository
```bash
git clone https://github.com/Binary-Insights/Assignment_03.git
cd Assignment_03
```

---

### 3. Configure Environment
Create a `.env` file in the project root:

```env
# AWS Configuration (for S3 integration)
AWS_ACCESS_KEY_ID=your-aws-access-key-id-here
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key-here
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# LangSmith Configuration (for tracing)
LANGSMITH_API_KEY=your-langsmith-api-key-here

# Pinecone Configuration (Vector Database)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=bigdata-assignment-03
PINECONE_EMBEDDING_DIMENSION=3072
PINECONE_EMBEDDING_MODEL=text-embedding-3-large

# Project Configuration
PROJECT_NAME=Assignment_03
LOG_LEVEL=INFO

# PostgreSQL Configuration (Concept Caching)
DB_HOST=172.24.98.166
DB_PORT=5432
DB_NAME=concept_db
DB_USER=postgres
DB_PASSWORD=your-postgres-password-here
```

---

### 4. Build and Start Airflow and Services
```bash
# Initialize Airflow database (first time only)
docker compose up airflow-init

# Start all containers
docker compose up -d
```

---

### 5. Access Dashboards
| Service | URL | Default Credentials |
|:--|:--|:--|
| Airflow UI | http://localhost:8080 | `airflow / airflow` |
| Streamlit UI | http://localhost:8501 | — |
| FastAPI Docs | http://localhost:8000/docs | — |

---

## Run Instructions

### 1. Full Pipeline (Recommended)
Run the complete workflow directly from **Airflow UI** by triggering the DAGs:
- `src/dags/fintbx_ingest_dag.py`
- `src/dags/financial_terms_enrichment_dag.py`

### 2. Manual Step-by-Step
1. **Split PDFs** → `split_pdf_task`  
2. **Parse with Docling** → `docling_parse_task`  
3. **Chunk Text** → `langchain_chunk_task`  
4. **Generate Embeddings** → `embedding_task`  
5. **Upsert Vectors to Pinecone** → `upsert_task`  
6. **Query via Streamlit/FastAPI`**


![Architecture Diagram](setup/architecture_diagram.png)

## Folder Structure
```
data/
├── raw/
│   └── fintbx.pdf
├── parsed/
│   ├── part_001.json
│   ├── part_002.json
│   └── ...
├── embeddings/
│   └── pinecone_vectors/
├── dags/
│   ├── fintbx_ingest_dag.py
│   └── concept_seed_dag.py
├── services/
│   ├── fastapi_app/
│   └── streamlit_ui/
└── logs/
    └── airflow/
```

---

## Troubleshooting
| Issue | Resolution |
|:--|:--|
| **AWS credentials not found** | Verify `.env` and restart Docker containers. |
| **OpenAI API error** | Check key and usage limits in `.env`. |
| **Pinecone connection failed** | Confirm index name and region exist. |
| **Docling parse error** | Ensure Python dependencies match Docling 2.57 requirements. |
| **“Dag references non-existent pools”** | Create pool in Airflow UI → Admin → Pools or remove the pool reference. |


## Codelabs – Documentation
Access interactive documentation and tutorials for Project AURELIA:

[**Automating Financial Concept Extraction Pipeline**](https://codelabs-preview.appspot.com/?file_id=11KEFuXQkCkHAY6lqDIBsLace-moi5qR07OULxfLJguw#7)  
[**Video Demo**]()

---

## Attestation
WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.

---

