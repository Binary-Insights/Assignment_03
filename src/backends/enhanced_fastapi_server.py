"""
Enhanced FastAPI Backend with Wikipedia Fallback and Structured Output
Integrates vector search, Wikipedia fallback, and instructor-based structuring
"""

import os
import warnings
import logging
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.enhanced_rag_pipeline import EnhancedRAGPipeline
from src.rag.instructor_models import FinancialConceptNote

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_VERBOSE'] = 'false'
os.environ['LANGCHAIN_DEBUG'] = 'false'

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced MATLAB RAG Assistant API",
    description="RAG backend with Wikipedia fallback and structured concept extraction",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline globally
pipeline = None


# ==================== Pydantic Models ====================

class RelevanceCheckRequest(BaseModel):
    """Request model for query relevance check"""
    query: str


class RelevanceCheckResponse(BaseModel):
    """Response model for query relevance check"""
    is_relevant: bool
    reason: str
    confidence: float


class EnhancedQueryRequest(BaseModel):
    """Request model for enhanced RAG query"""
    query: str
    use_vector_db: bool = True
    use_wikipedia: bool = True
    num_vector_results: int = 1


class EnhancedQueryResponse(BaseModel):
    """Response model for enhanced RAG query"""
    query: str
    concept_found_in_vector: bool
    source: str  # "vector_db", "wikipedia", or "cache"
    cached: bool
    structured_note: dict  # FinancialConceptNote as dict
    pinecone_context: Optional[str] = None
    wikipedia_context: Optional[str] = None
    processing_time_ms: float


class ConceptInfo(BaseModel):
    """Information about a cached concept"""
    term: str
    source: str
    search_count: int
    created_at: str


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    try:
        logger.info("=" * 70)
        logger.info("Enhanced FastAPI RAG Server Starting...")
        logger.info("=" * 70)
        
        pipeline = EnhancedRAGPipeline()
        logger.info("✓ Enhanced RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close pipeline on shutdown"""
    global pipeline
    if pipeline:
        pipeline.close()
        logger.info("Pipeline closed")


# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enhanced MATLAB RAG Assistant API",
        "version": "2.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced MATLAB RAG Assistant API",
        "version": "2.0.0",
        "features": {
            "vector_search": "Pinecone vector database search",
            "wikipedia_fallback": "Fallback to Wikipedia if not in vector DB",
            "structured_output": "Instructor-based structured concept extraction",
            "caching": "PostgreSQL caching for financial concepts"
        },
        "endpoints": {
            "health": "/health (GET)",
            "check_relevance": "/check-relevance (POST) - Check query relevance to finance/economics",
            "query": "/query (POST) - Enhanced query with fallback",
            "concepts": "/concepts (GET) - Get cached concepts",
            "docs": "/docs (GET) - Interactive API documentation"
        }
    }


@app.post("/check-relevance", response_model=RelevanceCheckResponse)
async def check_relevance(request: RelevanceCheckRequest):
    """
    Check if a query is relevant to finance/economics/mathematics
    Uses LLM to make intelligent relevance determination
    
    Args:
        request: RelevanceCheckRequest with query text
    
    Returns:
        RelevanceCheckResponse with relevance decision and explanation
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Checking relevance for query: {request.query[:50]}...")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Use LLM to determine relevance
        from langchain_core.messages import HumanMessage
        
        relevance_prompt = f"""Evaluate if the following query is relevant to finance, economics, or mathematics (preferably related to finance/economics).

Query: "{request.query}"

Respond in JSON format ONLY with no additional text:
{{
    "is_relevant": true/false,
    "reason": "brief explanation of why it is/isn't relevant",
    "confidence": 0.0 to 1.0 (how confident you are in this assessment),
    "topics": ["list", "of", "detected", "topics"]
}}

Relevant domains include:
- Finance: investments, portfolios, trading, derivatives, bonds, stocks, etc.
- Economics: economic analysis, markets, monetary/fiscal policy, supply/demand, etc.
- Mathematics for finance: statistics, probability, optimization, time series analysis, etc.
- MATLAB in finance: using MATLAB for financial analysis or computations
- Related: quantitative analysis, risk management, asset management, etc.

Irrelevant examples would be: cooking recipes, sports, history, entertainment, health (unless finance-related), etc.
Be fair but strict - the query should have a clear connection to at least one of the relevant domains."""
        
        response = pipeline.llm.invoke([HumanMessage(content=relevance_prompt)])
        response_text = response.content.strip()
        
        logger.info(f"LLM Relevance Response: {response_text[:200]}...")
        
        # Parse JSON response
        import json
        try:
            # Find JSON in response (in case LLM adds extra text)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Fallback: LLM didn't return valid JSON, use a safe default
                logger.warning("LLM response was not valid JSON, using fallback")
                return RelevanceCheckResponse(
                    is_relevant=False,
                    reason="Could not parse relevance check. Please try a clearer query.",
                    confidence=0.5
                )
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            is_relevant = result.get("is_relevant", False)
            reason = result.get("reason", "Could not determine relevance")
            confidence = result.get("confidence", 0.5)
            
            if is_relevant:
                logger.info(f"✓ Query is relevant: {reason}")
            else:
                logger.info(f"✗ Query is NOT relevant: {reason}")
            
            return RelevanceCheckResponse(
                is_relevant=is_relevant,
                reason=reason,
                confidence=float(confidence)
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON response: {e}")
            return RelevanceCheckResponse(
                is_relevant=False,
                reason="Could not evaluate query relevance. Please try again.",
                confidence=0.0
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking relevance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/query", response_model=EnhancedQueryResponse)
async def enhanced_query(request: EnhancedQueryRequest):
    """
    Enhanced query endpoint with Wikipedia fallback
    
    Flow:
    1. Search Pinecone vector DB for the concept
    2. If not found, check PostgreSQL cache
    3. If not cached, search Wikipedia
    4. Generate structured note using instructor model
    5. Cache the result in PostgreSQL
    
    Args:
        request: EnhancedQueryRequest with query parameters
    
    Returns:
        EnhancedQueryResponse with structured concept note
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Processing enhanced query: {request.query[:50]}...")
        
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query with enhanced pipeline
        response = pipeline.process_query(
            query=request.query,
            use_vector_db=request.use_vector_db,
            num_vector_results=request.num_vector_results
        )
        
        # Convert to response model
        return EnhancedQueryResponse(
            query=response.query,
            concept_found_in_vector=response.concept_found_in_vector,
            source=response.source,
            cached=response.cached,
            structured_note=response.structured_note.model_dump(),
            pinecone_context=response.pinecone_context,
            wikipedia_context=response.wikipedia_context[:500] if response.wikipedia_context else None,
            processing_time_ms=response.processing_time_ms
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/concepts")
async def get_cached_concepts(limit: int = 50):
    """
    Get cached financial concepts
    
    Args:
        limit: Maximum number of concepts to return
    
    Returns:
        List of cached concepts with metadata
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not pipeline.db:
        raise HTTPException(status_code=503, detail="Database not available - caching disabled")
    
    try:
        concepts = pipeline.db.get_all_concepts(limit=limit)
        
        return {
            "total": len(concepts),
            "concepts": [
                ConceptInfo(
                    term=c['term'],
                    source=c['wikipedia_source'],
                    search_count=c['search_count'],
                    created_at=c['created_at'].isoformat() if c['created_at'] else None
                )
                for c in concepts
            ]
        }
    
    except Exception as e:
        logger.error(f"Error retrieving concepts: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving concepts")


@app.get("/concept/{term}")
async def get_concept_detail(term: str):
    """
    Get detailed information about a cached concept
    
    Args:
        term: The financial concept term
    
    Returns:
        Detailed concept information with structured note
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not pipeline.db:
        raise HTTPException(status_code=503, detail="Database not available - caching disabled")
    
    try:
        concept = pipeline.db.get_concept(term)
        
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept not found: {term}")
        
        return {
            "term": concept['term'],
            "source": concept['wikipedia_source'],
            "created_at": concept['created_at'],
            "search_count": concept['search_count'],
            "structured_note": concept['structured_note']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving concept: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving concept")


@app.get("/stats")
async def get_statistics():
    """Get statistics about cached concepts and queries"""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not pipeline.db:
        raise HTTPException(status_code=503, detail="Database not available - caching disabled")
    
    try:
        concepts = pipeline.db.get_all_concepts(limit=1000)
        total_searches = sum(c['search_count'] for c in concepts)
        
        return {
            "total_cached_concepts": len(concepts),
            "total_searches": total_searches,
            "average_searches_per_concept": total_searches / len(concepts) if concepts else 0,
            "top_searched_concepts": [
                {
                    "term": c['term'],
                    "searches": c['search_count']
                }
                for c in sorted(concepts, key=lambda x: x['search_count'], reverse=True)[:10]
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Error getting statistics")


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn
    
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
