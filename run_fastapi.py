#!/usr/bin/env python
"""
Wrapper script to run the FastAPI server
Run from project root: python run_fastapi.py
"""
import sys
import os

# Add project root to path so imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import and run
from src.backends.enhanced_fastapi_server import app
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Enhanced MATLAB RAG FastAPI Server")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Pinecone vector search")
    print("  - Wikipedia fallback")
    print("  - Instructor-structured outputs")
    print("  - PostgreSQL concept caching")
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )
