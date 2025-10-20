"""
PostgreSQL Database Schema and Operations
Manages concept caching for financial terms
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import Json
import logging

logger = logging.getLogger(__name__)


class ConceptDatabase:
    """PostgreSQL database for caching financial concepts"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "concept_db",
                 user: str = "postgres",
                 password: str = "postgres"):
        """Initialize database connection"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to PostgreSQL database: {self.database}")
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Closed PostgreSQL connection")
    
    def initialize_schema(self):
        """Create tables if they don't exist"""
        try:
            self.cursor.execute("""
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
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    source VARCHAR(50),
                    concept_id INT REFERENCES financial_concepts(id),
                    response JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
            logger.info("Database schema initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Error initializing schema: {e}")
            self.conn.rollback()
            raise
    
    def insert_concept(self, 
                      term: str, 
                      wikipedia_content: str, 
                      wikipedia_source: str,
                      structured_note: Dict[str, Any]) -> int:
        """Insert a new financial concept"""
        try:
            self.cursor.execute("""
                INSERT INTO financial_concepts 
                (term, wikipedia_source, wikipedia_content, structured_note, search_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (term) DO UPDATE
                SET wikipedia_content = EXCLUDED.wikipedia_content,
                    structured_note = EXCLUDED.structured_note,
                    updated_at = CURRENT_TIMESTAMP,
                    search_count = financial_concepts.search_count + 1
                RETURNING id
            """, (term, wikipedia_source, wikipedia_content, Json(structured_note), 1))
            
            concept_id = self.cursor.fetchone()[0]
            self.conn.commit()
            logger.info(f"Inserted/Updated concept: {term} (ID: {concept_id})")
            return concept_id
        except psycopg2.Error as e:
            logger.error(f"Error inserting concept: {e}")
            self.conn.rollback()
            raise
    
    def get_concept(self, term: str) -> Optional[Dict[str, Any]]:
        """Retrieve a concept from cache"""
        try:
            self.cursor.execute("""
                SELECT id, term, wikipedia_source, wikipedia_content, structured_note, 
                       created_at, search_count
                FROM financial_concepts
                WHERE term = %s
            """, (term,))
            
            result = self.cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'term': result[1],
                    'wikipedia_source': result[2],
                    'wikipedia_content': result[3],
                    'structured_note': result[4],
                    'created_at': result[5],
                    'search_count': result[6]
                }
            return None
        except psycopg2.Error as e:
            logger.error(f"Error retrieving concept: {e}")
            return None
    
    def log_query(self, 
                  query: str, 
                  source: str, 
                  concept_id: int,
                  response: Dict[str, Any]):
        """Log a query operation"""
        try:
            self.cursor.execute("""
                INSERT INTO query_logs (query, source, concept_id, response)
                VALUES (%s, %s, %s, %s)
            """, (query, source, concept_id, Json(response)))
            
            self.conn.commit()
        except psycopg2.Error as e:
            logger.error(f"Error logging query: {e}")
            self.conn.rollback()
    
    def get_all_concepts(self, limit: int = 100) -> list:
        """Get all cached concepts"""
        try:
            self.cursor.execute("""
                SELECT id, term, wikipedia_source, structured_note, 
                       created_at, search_count
                FROM financial_concepts
                ORDER BY search_count DESC, updated_at DESC
                LIMIT %s
            """, (limit,))
            
            results = self.cursor.fetchall()
            concepts = []
            for result in results:
                concepts.append({
                    'id': result[0],
                    'term': result[1],
                    'wikipedia_source': result[2],
                    'structured_note': result[3],
                    'created_at': result[4],
                    'search_count': result[5]
                })
            return concepts
        except psycopg2.Error as e:
            logger.error(f"Error retrieving concepts: {e}")
            return []
    
    def update_search_count(self, term: str):
        """Update search count for a concept"""
        try:
            self.cursor.execute("""
                UPDATE financial_concepts
                SET search_count = search_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE term = %s
            """, (term,))
            
            self.conn.commit()
        except psycopg2.Error as e:
            logger.error(f"Error updating search count: {e}")
            self.conn.rollback()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    db = ConceptDatabase()
    try:
        db.connect()
        db.initialize_schema()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()
