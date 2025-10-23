"""
Enhanced RAG Pipeline with Wikipedia Fallback and Instructor Structuring
Orchestrates Pinecone search, Wikipedia fallback, and structured note generation
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pinecone import Pinecone as PineconeClient
import instructor
from openai import OpenAI

from src.rag.wikipedia_tools import get_tools, get_wikipedia_full_content
from concept_database import ConceptDatabase
from src.rag.instructor_models import FinancialConceptNote, QueryResponse
from src.rag.langsmith_config import LangSmithConfig, TraceContext

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure LangSmith tracing
LangSmithConfig.configure_langsmith()


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with Wikipedia fallback and structured output"""
    
    def __init__(self):
        """Initialize the enhanced RAG pipeline"""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Initialize instructor with OpenAI client for structured outputs
        # Create base OpenAI client
        base_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize instructor - this wraps the OpenAI client
        self.instructor_client = instructor.from_openai(base_openai_client)
        
        # Note: The instructor client uses the OpenAI client which will be auto-traced by LangSmith
        # if LANGSMITH_TRACING is enabled
        
        # Initialize Pinecone
        pinecone_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        pc = PineconeClient(api_key=pinecone_key)
        self.pinecone_index = pc.Index(index_name)
        
        # Initialize database
        try:
            # For Docker: use "postgres" database first to create "concept_db" if needed
            # For local: use direct concept_db connection
            db_host = os.getenv("DB_HOST", "postgres")
            db_port = int(os.getenv("DB_PORT", 5432))
            db_name = os.getenv("DB_NAME", "concept_db")
            db_user = os.getenv("DB_USER", "airflow")
            db_password = os.getenv("DB_PASSWORD", "airflow")
            
            logger.info(f"Connecting to database: {db_user}@{db_host}:{db_port}/{db_name}")
            
            self.db = ConceptDatabase(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password
            )
            self.db.connect()
            self.db.initialize_schema()
            logger.info("✓ Database connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}")
            logger.warning("⚠️ Running in limited mode without caching. Cache features will be disabled.")
            self.db = None
        
        # Get Wikipedia tools
        self.wikipedia_tools = get_tools()
    
    def search_vector_db(self, query: str, top_k: int = 5) -> Tuple[bool, Optional[str]]:
        """Search Pinecone vector database"""
        with TraceContext("search_vector_db", {"query": query, "top_k": top_k}):
            try:
                logger.info(f"Searching vector DB for: {query}")
                embedding = self.embeddings.embed_query(query)
                logger.info(f"[TRACE] Generated embedding for query (dimension: {len(embedding)})")
                
                results = self.pinecone_index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace="production"
                )
                
                matches = results.get('matches', [])
                if matches:
                    # Log match scores for tracing
                    for i, match in enumerate(matches):
                        logger.info(f"[TRACE] Match {i+1}: score={match.get('score', 0):.4f}")
                    
                    # Extract context from matches
                    context_text = matches[0].get('metadata', {}).get('text', '')
                    logger.info(f"Found in vector DB: {len(matches)} results")
                    return True, context_text, matches  # Return all matches
                
                logger.info("No results found in vector DB")
                return False, None, []
            
            except Exception as e:
                logger.error(f"Error searching vector DB: {e}")
                return False, None, []
    
    def check_database_cache(self, concept_term: str) -> Optional[Dict[str, Any]]:
        """Check if concept exists in PostgreSQL cache"""
        if not self.db:
            return None
        
        try:
            cached = self.db.get_concept(concept_term)
            if cached:
                logger.info(f"Found in cache: {concept_term}")
                self.db.update_search_count(concept_term)
                return cached
            return None
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None
    
    def _combine_vector_results(self, matches: list, num_results: int = 1) -> str:
        """Combine multiple vector search results into a single context string"""
        if not matches:
            return ""
        
        # Limit to requested number of results
        selected_matches = matches[:num_results]
        logger.info(f"Combining {len(selected_matches)} vector results for context")
        
        combined_text = []
        for i, match in enumerate(selected_matches, 1):
            score = match.get('score', 0)
            text = match.get('metadata', {}).get('text', '')
            combined_text.append(f"[Result {i} - Score: {score:.4f}]\n{text}\n")
        
        return "\n---\n".join(combined_text)
    
    def search_wikipedia_fallback(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia as fallback when not in vector DB"""
        with TraceContext("search_wikipedia_fallback", {"query": query}):
            try:
                logger.info(f"Searching Wikipedia for: {query}")
                logger.info(f"[TRACE] Calling get_wikipedia_full_content tool")
                
                # Direct function call (not through LangChain tools)
                result = get_wikipedia_full_content(query=query)
                
                if result.get("success"):
                    logger.info(f"Found on Wikipedia: {result.get('title')}")
                    logger.info(f"[TRACE] Wikipedia search successful - title: {result.get('title')}, content_length: {len(result.get('content', ''))}")
                    
                    return {
                        "success": True,
                        "title": result.get('title'),
                        "url": result.get('url'),
                        "content": result.get('content'),
                        "summary": result.get('summary')
                    }
                
                logger.warning(f"Wikipedia search failed for: {query}")
                logger.info(f"[TRACE] Wikipedia search unsuccessful - no matching article found")
                return {"success": False}
            
            except Exception as e:
                logger.error(f"Error searching Wikipedia: {e}")
                logger.info(f"[TRACE] Wikipedia search error: {str(e)}")
                return {"success": False}
    
    def generate_structured_note(self, 
                                concept_term: str,
                                content: str,
                                source_url: Optional[str] = None) -> FinancialConceptNote:
        """Generate structured note using instructor with strict adherence to provided content"""
        with TraceContext("generate_structured_note", {"concept_term": concept_term, "content_length": len(content)}):
            try:
                logger.info(f"Generating structured note for: {concept_term}")
                
                # Strict prompt that enforces context adherence
                prompt = f"""TASK: Extract structured financial information ONLY from the provided content.

CRITICAL RULES:
1. Extract information DIRECTLY from the provided content below
2. Do NOT generate or infer information not present in the content
3. If information is missing from the content, use "Not found in provided content" or similar
4. The confidence_score MUST reflect how much relevant information was found in the content
   - 0.9-1.0: Content is highly relevant and comprehensive
   - 0.7-0.9: Content is relevant but somewhat incomplete
   - 0.5-0.7: Content has limited relevance to the concept
   - 0.3-0.5: Content is poorly relevant, should trigger fallback
   - 0.0-0.3: Content is not relevant, should trigger fallback to Wikipedia/cache
5. Be strict about relevance - if content doesn't closely match "{concept_term}", set low confidence

CONCEPT TO EXTRACT: {concept_term}

PROVIDED CONTENT:
==================
{content}
==================

INSTRUCTIONS FOR DEFINITION SECTION:
- Extract primary definition from content
- Include alternative definitions if mentioned
- For code_examples: Extract or identify MATLAB code snippets from the content that demonstrate this concept
  * Look for actual MATLAB code blocks, formulas, or algorithmic descriptions
  * If found, preserve them exactly as they appear
  * Format as: "% MATLAB: [code]"
  * Include at least 2 relevant MATLAB examples if available in content
  * If MATLAB-specific functions (like fFinancial Toolbox functions) are mentioned, extract those
- For example_explanation: Provide detailed context explaining:
  * What each MATLAB code example does
  * How it relates to the concept
  * What the expected MATLAB output or result would be
  * How to use it in MATLAB Financial Toolbox context
  * Real-world application context in financial analysis using MATLAB

FULL EXTRACTION GUIDELINES:
- Extract definition, characteristics, applications, and related concepts from the content above
- All information must come from the provided content only
- If a section has no relevant information in the content, indicate "Not available in provided content"
- Calculate confidence_score based on relevance and completeness of content
- If content is not relevant to "{concept_term}", set confidence_score between 0.3-0.5

Create a structured note with:
1. Primary definition (from content, not generated)
2. Code examples with explanations demonstrating the concept
3. Alternative definition if mentioned
4. Context from content
5. Key characteristics mentioned in content
6. Importance level based on content emphasis
7. Use cases mentioned in content
8. Industry examples if provided
9. MATLAB relevance if mentioned
10. Related concepts mentioned in the content
11. Confidence score reflecting content quality and relevance"""
                
                logger.info(f"Generating structured note with strict prompt for: {concept_term}")
                logger.info(f"[TRACE] Calling instructor model for structured extraction")
                logger.info(f"[TRACE] PROMPT:\n{prompt}")  # ◄─── LOG THE PROMPT FOR LANGSMITH
                
                # Use instructor to get structured output
                # The instructor client wraps OpenAI client, which IS auto-traced by LangSmith
                structured_note = self.instructor_client.messages.create(
                    model="gpt-4o",
                    response_model=FinancialConceptNote,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                logger.info(f"[TRACE] LLM RESPONSE RECEIVED - Processing output...")  # ◄─── LOG RESPONSE RECEIVED
                
                # Log confidence score for monitoring
                confidence = structured_note.confidence_score
                logger.info(f"Generated structured note for: {concept_term}, confidence: {confidence:.2f}")
                logger.info(f"[TRACE] Extracted fields: definition, characteristics, applications, related_concepts")
                logger.info(f"[TRACE] Confidence score: {confidence:.2f} - {'PASS' if confidence >= 0.5 else 'FAIL (will trigger fallback)'}")
                
                # Check if confidence is too low - might need fallback
                if confidence < 0.5:
                    logger.warning(f"Low confidence ({confidence:.2f}) for concept '{concept_term}' - might need fallback search")
                
                return structured_note
            
            except Exception as e:
                logger.error(f"Error generating structured note: {e}")
                raise
    
    def cache_concept(self,
                     concept_term: str,
                     wikipedia_url: str,
                     wikipedia_content: str,
                     structured_note: FinancialConceptNote):
        """Cache a concept in PostgreSQL"""
        if not self.db:
            logger.info(f"⚠️ Skipping cache: Database not available")
            return
        
        try:
            logger.info(f"Caching concept: {concept_term}")
            self.db.insert_concept(
                term=concept_term,
                wikipedia_content=wikipedia_content,
                wikipedia_source=wikipedia_url,
                structured_note=structured_note.model_dump()
            )
            logger.info(f"Cached concept: {concept_term}")
        except Exception as e:
            logger.error(f"Error caching concept: {e}")
    
    def process_query(self, 
                     query: str,
                     use_vector_db: bool = True,
                     num_vector_results: int = 1) -> QueryResponse:
        """
        Process a query with fallback to Wikipedia and caching
        
        Flow:
        1. Try to find in vector DB
        2. If not found, extract concept term and check cache
        3. If not cached, search Wikipedia
        4. Generate structured note using instructor
        5. Cache in PostgreSQL
        
        Args:
            query: The user's query
            use_vector_db: Whether to search vector DB
            num_vector_results: Number of vector DB results to use for structured note (1-10)
        """
        with TraceContext("process_query", {"query": query, "use_vector_db": use_vector_db, "num_vector_results": num_vector_results}):
            start_time = time.time()
            
            try:
                logger.info(f"Processing query: {query}")
                logger.info(f"Using {num_vector_results} vector result(s) for context")
                logger.info(f"[TRACE] Query routing: vector_db={use_vector_db}, cache=true, wikipedia=true")
                
                # Extract concept term from query
                concept_term = self._extract_concept_term(query)
                logger.info(f"Extracted concept term: {concept_term}")
                logger.info(f"[TRACE] Concept extraction complete")
                
                # Step 1: Try vector DB search
                vector_found = False
                vector_context = None
                vector_matches = []
                
                if use_vector_db:
                    logger.info(f"[TRACE] STEP 1: Attempting Vector DB search...")
                    vector_found, vector_context, vector_matches = self.search_vector_db(query)
                
                # Step 2: If found in vector DB, use that
                if vector_found and vector_matches:
                    logger.info("Using vector DB result")
                    logger.info(f"[TRACE] STEP 2: Vector DB found - generating structured note")
                    
                    # Combine multiple results if requested
                    combined_context = self._combine_vector_results(vector_matches, num_vector_results)
                    logger.info(f"[TRACE] Combined {num_vector_results} results into context")
                    
                    # Generate structured note from combined context
                    structured_note = self.generate_structured_note(
                        concept_term,
                        combined_context
                    )
                    
                    # CHECK CONFIDENCE QUALITY - if low, use fallback
                    confidence = structured_note.confidence_score
                    logger.info(f"[TRACE] Confidence check: {confidence:.2f}")
                    
                    if confidence < 0.5:
                        logger.warning(f"Vector DB result has low confidence ({confidence:.2f}), trying fallback...")
                        logger.info(f"[TRACE] FALLBACK TRIGGERED: Confidence < 0.5, proceeding to cache/Wikipedia")
                        # Fall through to cache/Wikipedia checks instead of returning low-quality result
                        vector_found = False  # Treat as not found, proceed to cache/Wikipedia
                    else:
                        # High confidence - return vector DB result
                        logger.info(f"[TRACE] QUALITY PASS: Confidence {confidence:.2f} >= 0.5, returning vector DB result")
                        processing_time = (time.time() - start_time) * 1000
                        
                        response = QueryResponse(
                            query=query,
                            concept_found_in_vector=True,
                            source="vector_db",
                            structured_note=structured_note,
                            pinecone_context=combined_context,
                            wikipedia_context=None,
                            cached=False,
                            processing_time_ms=processing_time
                        )
                        logger.info(f"[TRACE] RESPONSE: source=vector_db, confidence={confidence:.2f}, time={processing_time:.2f}ms")
                        return response
                
                # If we get here, either vector_db was not found or confidence was too low
                # So proceed to cache/Wikipedia checks
                logger.info(f"[TRACE] STEP 3: Vector DB not found or low confidence - checking cache...")
                
                # Step 3: Check PostgreSQL cache
                logger.info("Vector DB not found, checking cache...")
                cached_concept = self.check_database_cache(concept_term)
                
                if cached_concept:
                    logger.info(f"Found in cache: {concept_term}")
                    logger.info(f"[TRACE] CACHE HIT: Retrieved from PostgreSQL")
                    
                    # Retrieve structured note from cache
                    structured_note_dict = cached_concept.get('structured_note')
                    if structured_note_dict:
                        # Recreate the model from cache
                        structured_note = FinancialConceptNote(**structured_note_dict)
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        response = QueryResponse(
                            query=query,
                            concept_found_in_vector=False,
                            source="cache",
                            structured_note=structured_note,
                            wikipedia_context=cached_concept.get('wikipedia_content'),
                            cached=True,
                            processing_time_ms=processing_time
                        )
                        logger.info(f"[TRACE] RESPONSE: source=cache, confidence={structured_note.confidence_score:.2f}, time={processing_time:.2f}ms")
                        return response
                
                logger.info(f"[TRACE] STEP 4: Not in cache - searching Wikipedia...")
                
                # Step 4: Search Wikipedia
                logger.info("Not in cache, searching Wikipedia...")
                wikipedia_result = self.search_wikipedia_fallback(concept_term)
                logger.info(f"[TRACE] Wikipedia search result: success={wikipedia_result.get('success')}")
                
                if not wikipedia_result.get("success"):
                    logger.error(f"Could not find information about: {concept_term}")
                    raise ValueError(f"Could not find information about: {concept_term}")
                
                logger.info(f"[TRACE] STEP 5: Generating structured note from Wikipedia content...")
                
                # Step 5: Generate structured note
                structured_note = self.generate_structured_note(
                    concept_term,
                    wikipedia_result['content'],
                    wikipedia_result.get('url')
                )
                
                logger.info(f"[TRACE] STEP 6: Caching concept in PostgreSQL...")
                
                # Step 6: Cache in PostgreSQL
                self.cache_concept(
                    concept_term,
                    wikipedia_result.get('url', ''),
                    wikipedia_result['content'],
                    structured_note
                )
                
                # Log the query
                if self.db:
                    self.db.log_query(
                        query,
                        "wikipedia",
                        None,
                        structured_note.model_dump()
                    )
                
                processing_time = (time.time() - start_time) * 1000
                
                response = QueryResponse(
                    query=query,
                    concept_found_in_vector=False,
                    source="wikipedia",
                    structured_note=structured_note,
                    wikipedia_context=wikipedia_result['content'],
                    cached=False,
                    processing_time_ms=processing_time
                )
                logger.info(f"[TRACE] RESPONSE: source=wikipedia, confidence={structured_note.confidence_score:.2f}, time={processing_time:.2f}ms")
                return response
            
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise
    
    def _extract_concept_term(self, query: str) -> str:
        """Extract the main financial concept from a query"""
        try:
            # Use LLM to extract concept term
            prompt = f"""
            Extract the main financial concept or term from this query.
            Return only the term, nothing else.
            
            Query: {query}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            term = response.content.strip()
            return term
        except Exception as e:
            logger.warning(f"Error extracting concept term: {e}")
            # Fallback: use first few words
            return query.split()[:3].__str__()
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    # Example usage
    pipeline = EnhancedRAGPipeline()
    
    try:
        # Test query
        query = "What is portfolio optimization in finance?"
        result = pipeline.process_query(query)
        
        print("\n" + "="*70)
        print("Query Response:")
        print("="*70)
        print(f"Query: {result.query}")
        print(f"Source: {result.source}")
        print(f"Cached: {result.cached}")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")
        print("\nStructured Note:")
        print(result.structured_note.model_dump_json(indent=2))
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    finally:
        pipeline.close()
