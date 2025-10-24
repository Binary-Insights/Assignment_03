"""
Financial Terms Enrichment from Wikipedia
Reads financial terms from "Outline of Finance" Wikipedia page,
enriches them using LLM, and caches in PostgreSQL with provenance tracking.

Features:
- Reads "Outline of Finance" Wikipedia page for financial terms
- Caches unique terms in PostgreSQL
- Skips already processed terms (restart resilience)
- Tracks provenance (source, timestamp, status)
- Enriches terms with LLM understanding
- DAG-ready with logging and error handling
"""

import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import wikipedia
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class FinancialTerm:
    """Financial term with metadata"""
    term: str
    definition: str = ""
    wikipedia_content: str = ""
    llm_explanation: str = ""
    category: str = ""
    source_url: str = ""
    status: str = "pending"  # pending, processed, failed
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""
    processed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProcessingStatus:
    """Track processing status"""
    total_terms: int = 0
    processed_terms: int = 0
    failed_terms: int = 0
    skipped_terms: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Database Operations
# ============================================================================

class FinancialTermsDatabase:
    """PostgreSQL database operations for financial terms"""

    def __init__(self, host: str = None, port: int = None, database: str = None,
                 user: str = None, password: str = None):
        """Initialize database connection"""
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', 5432))
        self.database = database or os.getenv('DB_NAME', 'concept_db')
        self.user = user or os.getenv('DB_USER', 'airflow')
        self.password = password or os.getenv('DB_PASSWORD', 'airflow')
        
        self.conn = None
        self.connect()
        self.initialize_tables()

    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.user}@{self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def initialize_tables(self):
        """Create financial_terms table if not exists"""
        try:
            cursor = self.conn.cursor()
            
            # Create financial_terms table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_terms (
                    id SERIAL PRIMARY KEY,
                    term VARCHAR(255) UNIQUE NOT NULL,
                    definition TEXT,
                    wikipedia_content TEXT,
                    llm_explanation TEXT,
                    category VARCHAR(100),
                    source_url VARCHAR(500),
                    status VARCHAR(20) DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    provenance JSONB DEFAULT '{}'
                );
            """)
            
            # Create index on term for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_financial_terms_term 
                ON financial_terms(term);
            """)
            
            # Create index on status for filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_financial_terms_status 
                ON financial_terms(status);
            """)
            
            self.conn.commit()
            logger.info("✅ Database tables initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing tables: {e}")
            self.conn.rollback()
            raise

    def insert_or_update_term(self, term: FinancialTerm, provenance: Dict[str, Any] = None) -> bool:
        """Insert or update a financial term"""
        try:
            cursor = self.conn.cursor()
            
            provenance = provenance or {
                "source": "wikipedia_outline_of_finance",
                "created_at": datetime.now().isoformat(),
                "dag_run_id": os.getenv('AIRFLOW_RUN_ID', 'manual'),
                "task_id": os.getenv('AIRFLOW_TASK_ID', 'financial_terms_enrichment')
            }
            
            cursor.execute("""
                INSERT INTO financial_terms 
                (term, definition, wikipedia_content, llm_explanation, category, 
                 source_url, status, error_message, created_at, updated_at, provenance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (term) DO UPDATE SET
                    definition = COALESCE(EXCLUDED.definition, financial_terms.definition),
                    wikipedia_content = COALESCE(EXCLUDED.wikipedia_content, financial_terms.wikipedia_content),
                    llm_explanation = COALESCE(EXCLUDED.llm_explanation, financial_terms.llm_explanation),
                    category = COALESCE(EXCLUDED.category, financial_terms.category),
                    source_url = COALESCE(EXCLUDED.source_url, financial_terms.source_url),
                    status = COALESCE(EXCLUDED.status, financial_terms.status),
                    error_message = COALESCE(EXCLUDED.error_message, financial_terms.error_message),
                    updated_at = CURRENT_TIMESTAMP,
                    provenance = COALESCE(EXCLUDED.provenance, financial_terms.provenance);
            """, (
                term.term,
                term.definition,
                term.wikipedia_content,
                term.llm_explanation,
                term.category,
                term.source_url,
                term.status,
                term.error_message,
                term.created_at or datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(provenance)
            ))
            
            self.conn.commit()
            logger.info(f"✅ Stored term: {term.term}")
            return True
        except Exception as e:
            logger.error(f"❌ Error storing term {term.term}: {e}")
            self.conn.rollback()
            return False

    def get_pending_terms(self, limit: int = None) -> List[FinancialTerm]:
        """Get pending terms that haven't been processed"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            query = "SELECT * FROM financial_terms WHERE status = 'pending' ORDER BY created_at"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            terms = []
            for row in rows:
                term = FinancialTerm(
                    term=row['term'],
                    definition=row['definition'] or "",
                    wikipedia_content=row['wikipedia_content'] or "",
                    llm_explanation=row['llm_explanation'] or "",
                    category=row['category'] or "",
                    source_url=row['source_url'] or "",
                    status=row['status'],
                    error_message=row['error_message'] or "",
                    created_at=row['created_at'].isoformat() if row['created_at'] else "",
                    updated_at=row['updated_at'].isoformat() if row['updated_at'] else "",
                    processed_at=row['processed_at'].isoformat() if row['processed_at'] else ""
                )
                terms.append(term)
            
            logger.info(f"Found {len(terms)} pending terms")
            return terms
        except Exception as e:
            logger.error(f"❌ Error fetching pending terms: {e}")
            return []

    def get_term(self, term: str) -> Optional[FinancialTerm]:
        """Get a specific term"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM financial_terms WHERE term = %s", (term,))
            row = cursor.fetchone()
            
            if row:
                return FinancialTerm(
                    term=row['term'],
                    definition=row['definition'] or "",
                    wikipedia_content=row['wikipedia_content'] or "",
                    llm_explanation=row['llm_explanation'] or "",
                    category=row['category'] or "",
                    source_url=row['source_url'] or "",
                    status=row['status'],
                    error_message=row['error_message'] or ""
                )
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching term {term}: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
                FROM financial_terms;
            """)
            row = cursor.fetchone()
            return {
                "total": row['total'] or 0,
                "processed": row['processed'] or 0,
                "failed": row['failed'] or 0,
                "pending": row['pending'] or 0
            }
        except Exception as e:
            logger.error(f"❌ Error fetching stats: {e}")
            return {"total": 0, "processed": 0, "failed": 0, "pending": 0}

    def get_total_wikipedia_terms(self) -> int:
        """Get total number of terms extracted from Wikipedia (for batch calculation)"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT COUNT(*) as total FROM financial_terms;")
            row = cursor.fetchone()
            return row['total'] or 0
        except Exception as e:
            logger.error(f"❌ Error counting terms: {e}")
            return 0

    def get_extraction_progress(self) -> Dict[str, Any]:
        """Get extraction progress for incremental extraction"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT 
                    COUNT(*) as extracted_count,
                    MAX(created_at) as last_extraction_time
                FROM financial_terms;
            """)
            row = cursor.fetchone()
            return {
                "extracted_count": row['extracted_count'] or 0,
                "last_extraction_time": row['last_extraction_time'].isoformat() if row['last_extraction_time'] else None
            }
        except Exception as e:
            logger.error(f"❌ Error getting extraction progress: {e}")
            return {"extracted_count": 0, "last_extraction_time": None}

    def update_term_status(self, term: str, status: str, explanation: str = "", error: str = "") -> bool:
        """Update term processing status"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE financial_terms 
                SET status = %s, 
                    llm_explanation = %s,
                    error_message = %s,
                    processed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE term = %s;
            """, (status, explanation, error, term))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error updating term status: {e}")
            self.conn.rollback()
            return False

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✅ Database connection closed")


# ============================================================================
# Wikipedia Operations
# ============================================================================

class WikipediaFinanceTermsExtractor:
    """Extract financial terms from Wikipedia"""

    def __init__(self):
        self.source_page = "Outline of finance"
        self.source_url = f"https://en.wikipedia.org/wiki/{self.source_page.replace(' ', '_')}"

    def extract_terms(self) -> List[Tuple[str, str]]:
        """
        Extract financial terms from "Outline of Finance" Wikipedia page
        Returns list of (term, category) tuples
        """
        try:
            logger.info(f"🔍 Fetching Wikipedia page: {self.source_page}")
            page = wikipedia.page(self.source_page, auto_suggest=False)
            content = page.content
            
            # Extract terms from headers and content
            terms = []
            sections = content.split('\n\n')
            
            current_category = "General"
            for section in sections:
                lines = section.split('\n')
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Check if line is a subsection header (marked with === or ==)
                    if line.startswith('=='):
                        current_category = line.replace('=', '').strip()
                        continue
                    
                    # Only extract bullet points or simple single-line terms
                    term = line.strip()
                    
                    # Skip headers, long sentences, and special markers
                    skip_patterns = [
                        '==', '===', '[', 'Contents', 'See also', 'References',
                        'the ', 'The ', 'and ', 'or ', 'is ', 'are ', 'as '
                    ]
                    
                    # Skip if matches patterns or is too short/long/contains certain markers
                    if any(term.startswith(p) for p in skip_patterns):
                        continue
                    
                    # Skip if it's a full sentence (contains multiple spaces suggesting multiple words)
                    word_count = len(term.split())
                    if word_count > 5 or word_count < 1:
                        continue
                    
                    # Skip if contains special characters that aren't part of financial terms
                    if term.count('–') > 1 or term.count('-') > 2:
                        continue
                    
                    # Add the term
                    if len(term) > 2:  # Minimum length
                        terms.append((term, current_category))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term, category in terms:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append((term, category))
            
            logger.info(f"✅ Extracted {len(unique_terms)} unique financial terms from Wikipedia")
            return unique_terms
        except Exception as e:
            logger.error(f"❌ Error extracting terms from Wikipedia: {e}")
            return []

    def get_term_definition(self, term: str) -> Optional[Dict[str, str]]:
        """Get definition and content for a specific term"""
        try:
            logger.info(f"📖 Searching Wikipedia for: {term}")
            search_results = wikipedia.search(term, results=1)
            
            if not search_results:
                logger.warning(f"⚠️ No Wikipedia results for: {term}")
                return None
            
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                summary = page.summary[:500] if page.summary else ""
                content = page.content[:2000] if page.content else ""
                
                return {
                    "term": term,
                    "definition": summary,
                    "content": content,
                    "url": page.url
                }
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = page.summary[:500] if page.summary else ""
                    content = page.content[:2000] if page.content else ""
                    return {
                        "term": term,
                        "definition": summary,
                        "content": content,
                        "url": page.url
                    }
                return None
            except wikipedia.exceptions.PageError:
                logger.warning(f"⚠️ Wikipedia page not found for: {term}")
                return None
        except Exception as e:
            logger.error(f"❌ Error getting term definition: {e}")
            return None


# ============================================================================
# LLM Enrichment
# ============================================================================

class FinancialTermEnricher:
    """Enrich financial terms with LLM"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o')

    def enrich_term(self, term: str, definition: str = "", wikipedia_content: str = "") -> str:
        """Enrich a financial term with LLM explanation"""
        try:
            logger.info(f"🧠 Enriching term with LLM: {term}")
            
            prompt = f"""
You are a financial expert. Explain the following financial term in a concise, clear way suitable for data analysis and finance professionals.

Financial Term: {term}

Wikipedia Summary: {definition[:300] if definition else "Not available"}

Provide a clear explanation covering:
1. Definition (1-2 sentences)
2. Key use cases or applications
3. Related concepts

Keep the explanation to 200-300 words maximum.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial expert providing clear, concise explanations of financial terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"✅ Enriched term: {term}")
            return explanation
        except Exception as e:
            logger.error(f"❌ Error enriching term {term}: {e}")
            raise

    def classify_term(self, term: str, definition: str = "") -> str:
        """Classify financial term into categories"""
        try:
            categories = [
                "Asset Management",
                "Banking",
                "Capital Markets",
                "Corporate Finance",
                "Derivatives",
                "Economics",
                "Fixed Income",
                "Foreign Exchange",
                "Insurance",
                "Investments",
                "Options & Futures",
                "Personal Finance",
                "Portfolio Theory",
                "Real Estate",
                "Risk Management",
                "Trading",
                "Other"
            ]
            
            prompt = f"""
Classify the following financial term into ONE of these categories:
{', '.join(categories)}

Financial Term: {term}
Definition: {definition[:200] if definition else ""}

Respond with ONLY the category name.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial classifier. Respond with only the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            category = response.choices[0].message.content.strip()
            logger.info(f"✅ Classified term {term}: {category}")
            return category
        except Exception as e:
            logger.error(f"❌ Error classifying term {term}: {e}")
            return "Other"


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_financial_terms(batch_size: int = 10, max_terms: int = None, incremental: bool = True):
    """
    Main pipeline to process financial terms
    
    Args:
        batch_size: Number of pending terms to process per run (base size)
        max_terms: Maximum total terms to process (for testing, None = no limit)
        incremental: If True, automatically increase batch size based on processing cycles
    """
    logger.info("=" * 80)
    logger.info("🚀 Financial Terms Enrichment Pipeline Started")
    logger.info("=" * 80)
    
    db = FinancialTermsDatabase()
    extractor = WikipediaFinanceTermsExtractor()
    enricher = FinancialTermEnricher()
    
    try:
        # Step 1: Get current stats
        stats = db.get_stats()
        progress = db.get_extraction_progress()
        
        logger.info(f"📊 Current database stats: {stats}")
        logger.info(f"📈 Extraction progress: {progress}")
        
        # Calculate incremental batch size if enabled
        actual_batch_size = batch_size
        if incremental and stats['total'] > 0:
            # Increase batch size based on number of processed terms
            # Formula: base_batch_size + (processed_count // 10)
            # This means: batch increases by 1 for every 10 processed terms
            incremental_increase = stats['processed'] // 10
            actual_batch_size = batch_size + incremental_increase
            logger.info(f"📈 Incremental batch size: {actual_batch_size} (base: {batch_size} + increase: {incremental_increase})")
        
        # Step 2: Extract new terms from Wikipedia (only if no terms exist)
        if stats['total'] == 0:
            logger.info("📝 Extracting new terms from Wikipedia...")
            wikipedia_terms = extractor.extract_terms()
            
            # Extract all available terms first
            terms_to_insert = wikipedia_terms[:max_terms] if max_terms else wikipedia_terms
            logger.info(f"📚 Found {len(terms_to_insert)} terms to extract initially")
            
            for term_text, category in terms_to_insert:
                # Check if term already exists
                existing = db.get_term(term_text)
                if existing:
                    logger.info(f"⏭️ Skipping existing term: {term_text}")
                    continue
                
                # Create new term record
                term = FinancialTerm(
                    term=term_text,
                    category=category,
                    status="pending",
                    source_url=extractor.source_url
                )
                
                provenance = {
                    "source": "wikipedia_outline_of_finance",
                    "created_at": datetime.now().isoformat(),
                    "extraction_method": "wikipedia_page_scraping",
                    "initial_extraction": True
                }
                
                db.insert_or_update_term(term, provenance)
            
            # Get updated stats after extraction
            stats = db.get_stats()
            logger.info(f"📊 Updated database stats after extraction: {stats}")
        
        # Step 3: Process pending terms with incremental batch size
        logger.info(f"⏳ Processing up to {actual_batch_size} pending terms (incremental batch)...")
        pending_terms = db.get_pending_terms(limit=actual_batch_size)
        
        if not pending_terms:
            logger.info("✅ No pending terms to process")
        
        processed_count = 0
        for term_obj in pending_terms:
            try:
                logger.info(f"\n--- Processing: {term_obj.term} ---")
                
                # Get Wikipedia definition
                wiki_data = extractor.get_term_definition(term_obj.term)
                if wiki_data:
                    term_obj.definition = wiki_data['definition']
                    term_obj.wikipedia_content = wiki_data['content']
                    term_obj.source_url = wiki_data['url']
                    logger.info(f"✅ Retrieved Wikipedia definition")
                else:
                    logger.warning(f"⚠️ No Wikipedia definition found")
                
                # Enrich with LLM
                try:
                    explanation = enricher.enrich_term(
                        term_obj.term,
                        term_obj.definition,
                        term_obj.wikipedia_content
                    )
                    term_obj.llm_explanation = explanation
                    
                    # Classify term
                    category = enricher.classify_term(term_obj.term, term_obj.definition)
                    term_obj.category = category
                    
                    term_obj.status = "processed"
                    term_obj.processed_at = datetime.now().isoformat()
                    
                    db.update_term_status(
                        term_obj.term,
                        "processed",
                        explanation
                    )
                    
                    processed_count += 1
                    logger.info(f"✅ Successfully processed: {term_obj.term}")
                
                except Exception as e:
                    logger.error(f"❌ Error enriching term: {e}")
                    term_obj.status = "failed"
                    term_obj.error_message = str(e)
                    db.update_term_status(term_obj.term, "failed", error=str(e))
                    
            except Exception as e:
                logger.error(f"❌ Error processing term {term_obj.term}: {e}")
                db.update_term_status(term_obj.term, "failed", error=str(e))
        
        # Step 4: Final stats
        final_stats = db.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("📊 Final Statistics:")
        logger.info(f"   Total Terms: {final_stats['total']}")
        logger.info(f"   Processed: {final_stats['processed']}")
        logger.info(f"   Failed: {final_stats['failed']}")
        logger.info(f"   Pending: {final_stats['pending']}")
        logger.info(f"   Processed in this run: {processed_count}")
        logger.info(f"   Batch size used: {actual_batch_size}")
        
        if final_stats['pending'] > 0:
            logger.info(f"⏳ Next run will process up to {actual_batch_size + 1} terms (incremental growth)")
        
        logger.info("=" * 80)
        logger.info("✅ Financial Terms Enrichment Pipeline Completed")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "processed": processed_count,
            "batch_size": actual_batch_size,
            "stats": final_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        db.close()


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Financial Terms Enrichment Pipeline"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Base batch size for pending terms per run (default: 10)"
    )
    parser.add_argument(
        "-m", "--max-terms",
        type=int,
        default=None,
        metavar="N",
        help="Maximum total terms to process (for testing, None = no limit)"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental batch size growth (use fixed batch size)"
    )
    args = parser.parse_args()

    logger.info(f"Starting with batch_size={args.batch_size}, max_terms={args.max_terms}, incremental={not args.no_incremental}")
    result = process_financial_terms(
        batch_size=args.batch_size,
        max_terms=args.max_terms,
        incremental=not args.no_incremental
    )

    sys.exit(0 if result.get('success') else 1)
