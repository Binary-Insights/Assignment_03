"""
Airflow DAG for Financial Terms Enrichment
Extracts financial terms from "Outline of Finance" Wikipedia page,
enriches with LLM, and caches in PostgreSQL with provenance tracking.

Features:
- Restart resilient (continues from pending terms)
- Provenance tracking for audit trail
- Batch processing to handle large datasets
- Comprehensive logging and error handling
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, '/opt/airflow/workspace')

from src.enrichment.financial_terms_enrichment import (
    process_financial_terms,
    FinancialTermsDatabase
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Default DAG Arguments
# ============================================================================

default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 7),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=6),
}

# ============================================================================
# DAG Definition
# ============================================================================

dag = DAG(
    'financial_terms_enrichment_dag',
    default_args=default_args,
    description='Enrich financial terms from Wikipedia with LLM and cache in PostgreSQL (Hourly with incremental batch growth)',
    schedule_interval='0 * * * *',  # Run every hour (at minute 0 of every hour)
    catchup=False,
    tags=['data-enrichment', 'financial-terms', 'wikipedia', 'llm', 'hourly'],
)

# ============================================================================
# Task Functions
# ============================================================================

def check_database_connection(**context):
    """Verify database connection and show current stats"""
    try:
        logger.info("🔍 Checking database connection...")
        # Inside Docker, use 'postgres' as hostname (service name in docker-compose)
        db_host = os.getenv('DB_HOST')
        if not db_host or db_host == 'localhost':
            db_host = 'postgres'  # Use Docker service name
        
        db = FinancialTermsDatabase(
            host=db_host,
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'concept_db'),
            user=os.getenv('DB_USER', 'airflow'),
            password=os.getenv('DB_PASSWORD', 'airflow')
        )
        
        stats = db.get_stats()
        logger.info(f"✅ Database connected successfully")
        logger.info(f"   📊 Current Stats:")
        logger.info(f"      Total Terms: {stats['total']}")
        logger.info(f"      Processed: {stats['processed']}")
        logger.info(f"      Failed: {stats['failed']}")
        logger.info(f"      Pending: {stats['pending']}")
        
        db.close()
        
        # Push stats to XCom for downstream tasks
        context['task_instance'].xcom_push(key='db_stats', value=stats)
        
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise


def enrich_batch_terms(batch_size: int = 10, **context):
    """Process a batch of pending financial terms with incremental growth"""
    try:
        logger.info(f"🚀 Starting enrichment with base batch size of {batch_size} terms...")
        
        # Get DAG run info for provenance tracking
        dag_run_id = context['dag_run'].dag_id
        task_id = context['task'].task_id
        
        os.environ['AIRFLOW_RUN_ID'] = context['run_id']
        os.environ['AIRFLOW_TASK_ID'] = task_id
        
        # Process terms with incremental batch size enabled
        # Note: max_terms=None means NO LIMIT - extract all Wikipedia terms
        result = process_financial_terms(batch_size=batch_size, max_terms=None, incremental=True)
        
        if result['success']:
            actual_batch = result.get('batch_size', batch_size)
            logger.info(f"✅ Successfully processed {result['processed']} terms")
            logger.info(f"   Actual batch size used: {actual_batch}")
            logger.info(f"   Stats: {result['stats']}")
            context['task_instance'].xcom_push(key='enrichment_result', value=result)
            return result
        else:
            raise Exception(f"Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Enrichment batch failed: {e}")
        raise


def get_processing_status(**context):
    """Get final processing status"""
    try:
        logger.info("📊 Getting final processing status...")
        # Inside Docker, use 'postgres' as hostname (service name in docker-compose)
        db_host = os.getenv('DB_HOST')
        if not db_host or db_host == 'localhost':
            db_host = 'postgres'  # Use Docker service name
        
        db = FinancialTermsDatabase(
            host=db_host,
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'concept_db'),
            user=os.getenv('DB_USER', 'airflow'),
            password=os.getenv('DB_PASSWORD', 'airflow')
        )
        
        stats = db.get_stats()
        
        logger.info("\n" + "="*70)
        logger.info("📈 FINAL PROCESSING STATUS")
        logger.info("="*70)
        logger.info(f"Total Terms in Database: {stats['total']}")
        logger.info(f"✅ Processed: {stats['processed']}")
        logger.info(f"⏳ Pending: {stats['pending']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        
        if stats['pending'] > 0:
            logger.info(f"\n⚠️  {stats['pending']} terms still pending")
            logger.info("💡 Tip: Re-trigger this DAG to continue processing")
        else:
            logger.info("\n✅ All terms have been processed!")
        
        logger.info("="*70)
        
        db.close()
        
        # Push final stats to XCom
        context['task_instance'].xcom_push(key='final_stats', value=stats)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get status: {e}")
        raise


# ============================================================================
# Task Definitions
# ============================================================================

# Task 1: Check database connection
check_db = PythonOperator(
    task_id='check_database_connection',
    python_callable=check_database_connection,
    provide_context=True,
    dag=dag,
)

# Task 2: Enrich financial terms (batch of 5)
enrich_batch_1 = PythonOperator(
    task_id='enrich_batch_1',
    python_callable=enrich_batch_terms,
    op_kwargs={'batch_size': 5},
    provide_context=True,
    dag=dag,
)

# Task 3: Enrich more terms if pending (batch of 5)
enrich_batch_2 = PythonOperator(
    task_id='enrich_batch_2',
    python_callable=enrich_batch_terms,
    op_kwargs={'batch_size': 5},
    provide_context=True,
    dag=dag,
)

# Task 4: Get final status
final_status = PythonOperator(
    task_id='get_final_status',
    python_callable=get_processing_status,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# Task Dependencies
# ============================================================================

check_db >> enrich_batch_1 >> enrich_batch_2 >> final_status

# ============================================================================
# DAG Documentation
# ============================================================================

dag.doc_md = """
## Financial Terms Enrichment DAG

### Purpose
Extracts financial terms from the "Outline of Finance" Wikipedia page,
enriches them with LLM explanations, and caches in PostgreSQL with provenance tracking.

### Schedule
- **Frequency**: Every hour (at minute 0 of every hour)
- **Timezone**: UTC
- **First run**: On the hour (e.g., 12:00, 13:00, 14:00, etc.)
- **Incremental**: Batch size automatically increases with each run
- **Can be triggered manually** from Airflow UI anytime

### Features
- **Hourly Execution**: Automatically runs every hour
- **Unlimited Wikipedia Extraction**: Extracts ALL unique financial terms from Wikipedia (no limit)
- **Limited Processing**: Only processes 5 terms per hourly run
- **Incremental Batch Growth**: Batch size increases by 1 for every 10 processed terms
  * Hour 1: Process 5 terms
  * Hour 50: Process 6 terms (if 50+ processed)
  * Hour 3: Process 12 terms (if 20+ processed)
  * etc.
- **Restart Resilient**: Continues from where it left off if interrupted
- **Provenance Tracking**: Tracks source, timestamps, and processing status
- **LLM Enrichment**: Uses GPT-4o to provide clear explanations
- **Category Classification**: Automatically classifies terms into 17 financial categories

### Incremental Growth Logic
```
Batch Size = Base Batch (5) + (Processed Count // 10)

Examples:
- 0 processed terms     → batch size = 5
- 5 processed terms     → batch size = 5
- 10 processed terms    → batch size = 6
- 25 processed terms    → batch size = 7
- 100 processed terms   → batch size = 15
- 200 processed terms   → batch size = 25
```

This ensures controlled growth: the DAG processes 5+ terms hourly,
gradually expanding coverage while keeping costs low.

### Data Flow
```
Wikipedia "Outline of Finance"
    ↓
Extract Financial Terms (once at first run)
    ↓
PostgreSQL (pending)
    ↓
Hourly Schedule (every hour):
  1. Calculate incremental batch size
  2. For each pending term:
     - Get Wikipedia definition
     - Enrich with LLM explanation
     - Classify into category
     - Store in PostgreSQL (processed)
    ↓
Database: financial_terms table (continuously growing)
```

### Database Schema
```
financial_terms:
- id (SERIAL PRIMARY KEY)
- term (VARCHAR UNIQUE)
- definition (TEXT)
- wikipedia_content (TEXT)
- llm_explanation (TEXT)
- category (VARCHAR)
- source_url (VARCHAR)
- status (VARCHAR: pending, processed, failed)
- error_message (TEXT)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
- processed_at (TIMESTAMP)
- provenance (JSONB)
```

### Provenance Tracking
Each term includes provenance metadata:
```json
{
  "source": "wikipedia_outline_of_finance",
  "created_at": "2025-10-24T10:00:00",
  "dag_run_id": "financial_terms_enrichment_dag_20251024_100000",
  "task_id": "enrich_batch_1",
  "extraction_method": "wikipedia_page_scraping",
  "initial_extraction": true
}
```

### How to Monitor
1. **Airflow UI**: http://localhost:8080
   - View DAG execution history (refresh to see new hourly runs)
   - Check task logs for each run
   - Monitor incremental batch sizes growing over time
   
2. **Database Queries**:
```sql
-- View hourly progress
SELECT DATE_TRUNC('hour', processed_at) as hour, COUNT(*) as terms_processed 
FROM financial_terms 
WHERE status = 'processed' 
GROUP BY DATE_TRUNC('hour', processed_at) 
ORDER BY hour DESC 
LIMIT 24;

-- View batch size growth
SELECT status, COUNT(*) as count FROM financial_terms GROUP BY status;

-- View pending terms
SELECT term, category FROM financial_terms WHERE status = 'pending' LIMIT 10;
```

3. **Logs**: Check Airflow logs for each hourly run
   - View batch size used per hour
   - Monitor processing time per term
   - Check for any failures

### Performance Expectations
- **Per term**: ~30-60 seconds (Wikipedia fetch + LLM call)
- **Hour 1**: 5 terms ≈ 2.5-5 minutes
- **Hour 2**: 5 terms ≈ 2.5-5 minutes
- **Hour 50**: 6 terms ≈ 3-6 minutes
- **After 24 hours**: Processing 5-6 terms/hour (120-144 per day)
- **After 240 hours (10 days)**: Processing 14-15 terms/hour

### Cost Considerations
- Uses OpenAI GPT-4o API (~$0.015 per 1K input tokens)
- Wikipedia API is free
- PostgreSQL storage grows with terms (minimal impact)
- **Per run (5 terms)**: ~$0.0375
- **Per hour**: ~$0.0375
- **Per day (24 runs)**: ~$0.90
- **Per month**: ~$27
- **Per year**: ~$324

### Configuration
Edit in `.env`:
```
DB_HOST=postgres
DB_PORT=5432
DB_NAME=concept_db
DB_USER=airflow
DB_PASSWORD=airflow
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
```

### Restart Behavior
- If DAG stops/fails, it will resume on next scheduled hour
- DAG will automatically skip already processed terms
- Only pending terms will be processed with incremental batch size
- Progress is tracked in `financial_terms.status` field

### Manual Triggers
You can manually trigger this DAG anytime from Airflow UI:
1. Go to http://localhost:8080
2. Find "financial_terms_enrichment_dag"
3. Click the play button (▶)
4. DAG will use current incremental batch size

### Monitoring Queries
```sql
-- View all terms with status
SELECT term, category, status, processed_at FROM financial_terms ORDER BY processed_at DESC;

-- View extraction timeline
SELECT DATE(created_at) as date, COUNT(*) as new_terms_extracted 
FROM financial_terms 
GROUP BY DATE(created_at) 
ORDER BY date;

-- View hourly processing timeline (last 24 hours)
SELECT DATE_TRUNC('hour', processed_at) as hour, COUNT(*) as terms_processed 
FROM financial_terms 
WHERE status = 'processed' AND processed_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', processed_at) 
ORDER BY hour DESC;

-- View statistics by category
SELECT category, COUNT(*) as count, 
       SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed
FROM financial_terms 
GROUP BY category 
ORDER BY count DESC;

-- View terms needing enrichment
SELECT term, status, error_message FROM financial_terms WHERE status = 'failed' LIMIT 10;

-- View batch growth over time
SELECT 
    DATE_TRUNC('hour', processed_at) as hour,
    COUNT(*) as terms_this_hour,
    SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('hour', processed_at)) as cumulative_total
FROM financial_terms 
WHERE status = 'processed'
GROUP BY DATE_TRUNC('hour', processed_at)
ORDER BY hour DESC
LIMIT 24;
```
"""
