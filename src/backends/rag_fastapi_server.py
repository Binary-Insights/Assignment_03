"""
FastAPI Backend Server for MATLAB RAG Assistant
Handles all RAG operations (embeddings, search, answer generation)
This server should be run separately from the Streamlit frontend
Location: backends/rag_fastapi_server.py
"""

import os
import warnings
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Suppress all warnings and disable caching BEFORE any langchain imports
warnings.filterwarnings('ignore')
os.environ['LANGCHAIN_VERBOSE'] = 'false'
os.environ['LANGCHAIN_DEBUG'] = 'false'
os.environ['LANGCHAIN_CACHE'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Initialize langchain module FIRST before any LangChain imports
import langchain
langchain.verbose = False
langchain.debug = False
try:
    langchain.llm_cache = None
except:
    pass

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MATLAB RAG Assistant API",
    description="RAG backend for MATLAB Financial Toolbox queries",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted to specific domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str
    top_k: int = 5
    num_context: int = 3
    temperature: float = 0.0


class SourceMatch(BaseModel):
    """Individual source/match from Pinecone"""
    text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    query: str
    answer: str
    sources: list[SourceMatch]
    num_sources_retrieved: int
    model_used: str


class ConfigResponse(BaseModel):
    """Configuration and status response"""
    pinecone_configured: bool
    openai_configured: bool
    pinecone_index: Optional[str] = None
    models_available: list[str]


# ==================== Global State (Singleton Pattern) ====================

class RAGPipeline:
    """
    Singleton RAG pipeline manager
    Ensures efficient resource pooling for embeddings, LLM, and Pinecone
    """
    _instance = None
    _embeddings = None
    _llm = None
    _pinecone_index = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGPipeline, cls).__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Initialize embeddings, LLM, and Pinecone connection"""
        if self._embeddings is None:
            logger.info("Initializing OpenAI embeddings (text-embedding-3-large)...")
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        if self._llm is None:
            logger.info("Initializing ChatOpenAI LLM (gpt-4o)...")
            self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        if self._pinecone_index is None:
            pinecone_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            
            if not pinecone_key or not index_name:
                raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not configured in .env")
            
            logger.info(f"Connecting to Pinecone index: {index_name}")
            pc = Pinecone(api_key=pinecone_key)
            self._pinecone_index = pc.Index(index_name)
            logger.info("Successfully connected to Pinecone")
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self.initialize()
        return self._embeddings
    
    @property
    def llm(self):
        if self._llm is None:
            self.initialize()
        return self._llm
    
    @property
    def pinecone_index(self):
        if self._pinecone_index is None:
            self.initialize()
        return self._pinecone_index


# ==================== Helper Functions ====================

def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI"""
    pipeline = RAGPipeline()
    try:
        logger.info(f"Generating embedding for query (length: {len(text)})")
        embedding = pipeline.embeddings.embed_query(text)
        logger.info(f"Successfully generated embedding (dimensions: {len(embedding)})")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def search_pinecone(embedding: list, top_k: int = 5) -> dict:
    """Search Pinecone with embedding vector"""
    pipeline = RAGPipeline()
    try:
        logger.info(f"Searching Pinecone with top_k={top_k}")
        results = pipeline.pinecone_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="production"
        )
        num_matches = len(results.get('matches', []))
        logger.info(f"Pinecone search returned {num_matches} results")
        return results
    except Exception as e:
        logger.error(f"Error searching Pinecone: {e}")
        raise


def extract_context(results: dict, num_context: int) -> tuple[str, list]:
    """Extract context from Pinecone results"""
    context_parts = []
    context_items = []
    
    for match in results.get('matches', []):
        if 'metadata' in match and 'text' in match['metadata']:
            text = match['metadata']['text']
            context_parts.append(text)
            context_items.append(SourceMatch(
                text=text,
                score=match.get('score')
            ))
    
    context = "\n\n".join(context_parts[:num_context])
    logger.info(f"Extracted {len(context_parts[:num_context])} context sections out of {len(context_parts)} available")
    
    return context, context_items


def generate_answer(query: str, context: str, temperature: float = 0.0) -> str:
    """Generate answer using LLM with provided context"""
    pipeline = RAGPipeline()
    
    try:
        logger.info(f"Generating answer with temperature={temperature}")
        
        prompt_template = """Use the following pieces of context to answer the question at the end.
                        Answer only based on the context given.
                        If you don't know the answer, just say that you don't know.

                        CONTEXT:
                        {context}

                        QUESTION: {question}

                        ANSWER:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Update temperature if different from default
        if temperature != pipeline.llm.temperature:
            logger.info(f"Updating LLM temperature to {temperature}")
            pipeline._llm = ChatOpenAI(model="gpt-4o", temperature=temperature)
        
        response = pipeline.llm.invoke([HumanMessage(content=formatted_prompt)])
        answer = response.content
        
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on server startup"""
    try:
        logger.info("=" * 70)
        logger.info("FastAPI MATLAB RAG Server Starting...")
        logger.info("=" * 70)
        pipeline = RAGPipeline()
        pipeline.initialize()
        logger.info("✓ RAG pipeline initialized successfully on startup")
    except Exception as e:
        logger.warning(f"⚠ Warning during startup: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        dict: Status and service information
    """
    return {
        "status": "healthy",
        "service": "MATLAB RAG Assistant API",
        "version": "1.0.0"
    }


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get configuration and status
    
    Returns:
        ConfigResponse: Configuration details and availability
    """
    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    logger.info("Configuration requested")
    
    return ConfigResponse(
        pinecone_configured=bool(pinecone_key),
        openai_configured=bool(openai_key),
        pinecone_index=index_name,
        models_available=["text-embedding-3-large", "gpt-4o"]
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Main RAG query endpoint
    
    Process:
    1. Generate embedding for query
    2. Search Pinecone for relevant documents
    3. Extract context from results
    4. Generate answer using LLM
    
    Args:
        request: QueryRequest with query text and parameters
    
    Returns:
        QueryResponse: Answer with sources and metadata
    
    Raises:
        HTTPException: If validation fails or processing error occurs
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Validate input
        if not request.query.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.top_k < 1 or request.top_k > 100:
            logger.warning(f"Invalid top_k: {request.top_k}")
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        if request.num_context < 1 or request.num_context > request.top_k:
            logger.warning(f"Invalid num_context: {request.num_context} for top_k: {request.top_k}")
            raise HTTPException(status_code=400, detail="num_context must be between 1 and top_k")
        
        if not (0.0 <= request.temperature <= 1.0):
            logger.warning(f"Invalid temperature: {request.temperature}")
            raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 1.0")
        
        logger.info(f"✓ Input validation passed | top_k={request.top_k}, num_context={request.num_context}, temp={request.temperature}")
        
        # Step 1: Initialize pipeline
        pipeline = RAGPipeline()
        pipeline.initialize()
        
        # Step 2: Generate query embedding
        logger.info("Step 1/4: Generating query embedding...")
        query_embedding = generate_embedding(request.query)
        
        # Step 3: Search Pinecone
        logger.info("Step 2/4: Searching Pinecone...")
        results = search_pinecone(query_embedding, request.top_k)
        
        # Step 4: Extract context
        logger.info("Step 3/4: Extracting context...")
        context, context_items = extract_context(results, request.num_context)
        
        if not context:
            logger.warning("No relevant documents found in knowledge base")
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found in knowledge base"
            )
        
        # Step 5: Generate answer
        logger.info("Step 4/4: Generating answer...")
        answer = generate_answer(request.query, context, request.temperature)
        
        # Step 6: Return response
        logger.info("✓ Query processed successfully!")
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=context_items[:request.num_context],
            num_sources_retrieved=len(results.get('matches', [])),
            model_used="gpt-4o"
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/batch-query")
async def batch_query(queries: list[str]):
    """
    Batch query endpoint for processing multiple queries
    
    Args:
        queries: List of query strings
    
    Returns:
        dict: Results for each query
    """
    logger.info(f"Processing batch of {len(queries)} queries")
    results = []
    
    for i, query_text in enumerate(queries, 1):
        try:
            logger.info(f"Processing batch query {i}/{len(queries)}")
            request = QueryRequest(query=query_text)
            response = await query_rag(request)
            results.append({
                "query": query_text,
                "success": True,
                "response": response.dict()
            })
        except Exception as e:
            logger.error(f"Error processing batch query {i}: {e}")
            results.append({
                "query": query_text,
                "success": False,
                "error": str(e)
            })
    
    logger.info(f"Batch processing complete: {sum(1 for r in results if r['success'])}/{len(results)} successful")
    return {"results": results}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "MATLAB RAG Assistant API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation backend for MATLAB Financial Toolbox documentation",
        "endpoints": {
            "health": "/health (GET)",
            "config": "/config (GET)",
            "query": "/query (POST)",
            "batch_query": "/batch-query (POST)",
            "docs": "/docs (GET)",
            "openapi": "/openapi.json (GET)"
        },
        "instructions": "Start Streamlit frontend separately with: streamlit run src/frontend/rag_streamlit_frontend.py"
    }


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
    print("MATLAB RAG FastAPI Server")
    print("=" * 70)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation available at http://localhost:8000/docs")
    print("\nTo run Streamlit frontend separately:")
    print("  streamlit run src/frontend/rag_streamlit_frontend.py")
    print("\n" + "=" * 70 + "\n")
    
    # Run with: python -m uvicorn backends.rag_fastapi_server:app --reload --host localhost --port 8000
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )
