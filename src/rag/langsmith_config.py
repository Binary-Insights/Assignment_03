"""
LangSmith Configuration and Tracing Setup
Enables comprehensive tracing of the Enhanced RAG Pipeline behavior
"""

import os
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LangSmithConfig:
    """Configuration for LangSmith tracing"""
    
    # Environment variables for LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enhanced-rag-pipeline")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    # Tracing flags
    ENABLE_TRACING = os.getenv("ENABLE_LANGSMITH_TRACING", "true").lower() == "true"
    ENABLE_DEBUGGING = os.getenv("ENABLE_LANGSMITH_DEBUG", "false").lower() == "true"
    
    @staticmethod
    def configure_langsmith():
        """Configure LangSmith environment variables for tracing"""
        if not LangSmithConfig.LANGSMITH_API_KEY:
            logger.warning("LANGSMITH_API_KEY not set - tracing disabled")
            return False
        
        # Set environment variables for LangChain to auto-detect
        os.environ["LANGSMITH_API_KEY"] = LangSmithConfig.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = LangSmithConfig.LANGSMITH_PROJECT
        os.environ["LANGSMITH_ENDPOINT"] = LangSmithConfig.LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_TRACING"] = "true" if LangSmithConfig.ENABLE_TRACING else "false"
        
        if LangSmithConfig.ENABLE_DEBUGGING:
            os.environ["LANGSMITH_DEBUG"] = "true"
        
        logger.info(f"✓ LangSmith configured for project: {LangSmithConfig.LANGSMITH_PROJECT}")
        logger.info(f"  Tracing enabled: {LangSmithConfig.ENABLE_TRACING}")
        logger.info(f"  Debug mode: {LangSmithConfig.ENABLE_DEBUGGING}")
        
        return True


class TraceContext:
    """Context manager for LangSmith tracing"""
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize trace context
        
        Args:
            name: Name of the trace (e.g., "vector_db_search", "generate_structured_note")
            metadata: Additional metadata to attach to trace
        """
        self.name = name
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Enter trace context"""
        self.start_time = datetime.now()
        logger.info(f"[TRACE START] {self.name} - {self.start_time.isoformat()}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context"""
        self.end_time = datetime.now()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if exc_type:
            logger.error(f"[TRACE ERROR] {self.name} - {exc_type.__name__}: {exc_val}")
        else:
            logger.info(f"[TRACE END] {self.name} - Duration: {duration_ms:.2f}ms")


# Initialize LangSmith on module import
LangSmithConfig.configure_langsmith()


if __name__ == "__main__":
    print("LangSmith Configuration")
    print("=" * 60)
    print(f"API Key Set: {bool(LangSmithConfig.LANGSMITH_API_KEY)}")
    print(f"Project: {LangSmithConfig.LANGSMITH_PROJECT}")
    print(f"Endpoint: {LangSmithConfig.LANGSMITH_ENDPOINT}")
    print(f"Tracing Enabled: {LangSmithConfig.ENABLE_TRACING}")
    print(f"Debug Mode: {LangSmithConfig.ENABLE_DEBUGGING}")
