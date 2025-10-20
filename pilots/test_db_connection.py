#!/usr/bin/env python3
"""
Test script to verify Python can connect to PostgreSQL in WSL
"""

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_connection():
    """Test connection to PostgreSQL database"""
    try:
        # Get connection details from .env
        db_host = os.getenv('DB_HOST')
        db_port = int(os.getenv('DB_PORT', 5432))
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        
        print(f"Attempting to connect to PostgreSQL:")
        print(f"  Host: {db_host}")
        print(f"  Port: {db_port}")
        print(f"  Database: {db_name}")
        print(f"  User: {db_user}")
        print()
        
        # Connect to database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        print("✅ Connected successfully!")
        print()
        
        cursor = conn.cursor()
        
        # Test 1: Check financial_concepts table
        cursor.execute("SELECT COUNT(*) FROM financial_concepts;")
        financial_concepts_count = cursor.fetchone()[0]
        print(f"✅ financial_concepts table: {financial_concepts_count} records")
        
        # Test 2: Check query_logs table
        cursor.execute("SELECT COUNT(*) FROM query_logs;")
        query_logs_count = cursor.fetchone()[0]
        print(f"✅ query_logs table: {query_logs_count} records")
        
        # Test 3: Test insert (optional)
        cursor.execute("""
            INSERT INTO financial_concepts (term, wikipedia_source, search_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (term) DO UPDATE SET search_count = financial_concepts.search_count + 1
            RETURNING id, term, search_count;
        """, ('TEST_CONCEPT', 'test_source', 1))
        
        result = cursor.fetchone()
        print(f"✅ Insert test successful: {result}")
        
        conn.commit()
        
        # Test 4: List all tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\n✅ Tables in concept_db:")
        for table in tables:
            print(f"   - {table[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ All tests passed! Database connection is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed!")
        print(f"Error: {type(e).__name__}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if PostgreSQL is running in WSL: sudo service postgresql status")
        print("2. Verify .env file has correct DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        print("3. Verify the WSL IP address: hostname -I")
        print("4. Check PostgreSQL listen_addresses: sudo -u postgres psql -c 'SHOW listen_addresses;'")
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
