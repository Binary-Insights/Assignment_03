#!/usr/bin/env python3
"""
Database Initialization Script
Creates the concept_db database and schema on Docker startup
Designed to run before FastAPI/Airflow startup
"""

import os
import sys
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_admin_connection(host: str, port: int, user: str, password: str):
    """Get connection to postgres database (admin database)"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database="postgres",
            user=user,
            password=password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to postgres database: {e}")
        raise


def create_database(conn, db_name: str):
    """Create database if it doesn't exist"""
    try:
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"),
            [db_name]
        )
        
        if cursor.fetchone():
            logger.info(f"✓ Database '{db_name}' already exists")
            cursor.close()
            return
        
        # Create database
        cursor.execute(
            sql.SQL("CREATE DATABASE {} WITH ENCODING 'UTF8'").format(
                sql.Identifier(db_name)
            )
        )
        logger.info(f"✓ Created database '{db_name}'")
        cursor.close()
        
    except psycopg2.Error as e:
        logger.error(f"Error creating database: {e}")
        raise


def get_target_connection(host: str, port: int, database: str, user: str, password: str):
    """Get connection to target database"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to {database}: {e}")
        raise


def create_schema(conn):
    """Create tables in the database"""
    try:
        cursor = conn.cursor()
        
        # Create financial_concepts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_concepts (
                id SERIAL PRIMARY KEY,
                term VARCHAR(255) UNIQUE NOT NULL,
                wikipedia_source TEXT,
                wikipedia_content TEXT,
                structured_note JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_type VARCHAR(50) DEFAULT 'wikipedia',
                search_count INT DEFAULT 1
            )
        """)
        logger.info("✓ Created table 'financial_concepts'")
        
        # Create query_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                source VARCHAR(50),
                concept_id INT REFERENCES financial_concepts(id),
                response JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("✓ Created table 'query_logs'")
        
        # Create index on term for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_financial_concepts_term 
            ON financial_concepts(term)
        """)
        logger.info("✓ Created index on 'financial_concepts.term'")
        
        # Create index on search_count for sorting
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_financial_concepts_search_count 
            ON financial_concepts(search_count DESC)
        """)
        logger.info("✓ Created index on 'financial_concepts.search_count'")
        
        conn.commit()
        logger.info("✓ Schema created successfully")
        cursor.close()
        
    except psycopg2.Error as e:
        logger.error(f"Error creating schema: {e}")
        conn.rollback()
        raise


def verify_schema(conn):
    """Verify that all tables and indexes exist"""
    try:
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        logger.info(f"\n📊 Tables in database:")
        for table in tables:
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table[0]))
            )
            count = cursor.fetchone()[0]
            logger.info(f"   - {table[0]} ({count} rows)")
        
        # Check indexes
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY indexname
        """)
        
        indexes = cursor.fetchall()
        logger.info(f"\n📑 Indexes in database:")
        for index in indexes:
            logger.info(f"   - {index[0]}")
        
        cursor.close()
        
    except psycopg2.Error as e:
        logger.error(f"Error verifying schema: {e}")
        raise


def main():
    """Main initialization routine"""
    logger.info("="*70)
    logger.info("PostgreSQL Database Initialization Script")
    logger.info("="*70)
    
    # Get connection parameters from environment or use defaults
    # In Docker: use "postgres" service name, not localhost
    # In local dev: use localhost
    db_host = os.getenv("DB_HOST", "postgres")  # Changed default to "postgres" for Docker
    db_port = int(os.getenv("DB_PORT", 5432))
    db_name = os.getenv("DB_NAME", "concept_db")
    db_user = os.getenv("DB_USER", "airflow")
    db_password = os.getenv("DB_PASSWORD", "airflow")
    
    logger.info(f"\nConnection parameters:")
    logger.info(f"  Host: {db_host}")
    logger.info(f"  Port: {db_port}")
    logger.info(f"  Database: {db_name}")
    logger.info(f"  User: {db_user}")
    logger.info(f"  Password: {'*' * len(db_password) if db_password else 'None'}")
    
    try:
        # Step 1: Connect to admin database (postgres)
        logger.info(f"\n1️⃣ Connecting to admin database (postgres)...")
        admin_conn = get_admin_connection(db_host, db_port, db_user, db_password)
        logger.info("✓ Connected to admin database")
        
        # Step 2: Create concept_db database
        logger.info(f"\n2️⃣ Creating database '{db_name}'...")
        create_database(admin_conn, db_name)
        admin_conn.close()
        
        # Step 3: Connect to target database
        logger.info(f"\n3️⃣ Connecting to target database '{db_name}'...")
        target_conn = get_target_connection(db_host, db_port, db_name, db_user, db_password)
        logger.info(f"✓ Connected to '{db_name}'")
        
        # Step 4: Create schema
        logger.info(f"\n4️⃣ Creating schema...")
        create_schema(target_conn)
        
        # Step 5: Verify schema
        logger.info(f"\n5️⃣ Verifying schema...")
        verify_schema(target_conn)
        
        target_conn.close()
        
        logger.info("\n" + "="*70)
        logger.info("✅ Database initialization completed successfully!")
        logger.info("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Database initialization failed: {e}")
        logger.error("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
