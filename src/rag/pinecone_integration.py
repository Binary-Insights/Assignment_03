"""
Pinecone Integration for RAG System
Handles storage and retrieval of chunked documents with metadata
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime


@dataclass
class PineconeChunk:
    """Represents a chunk ready for Pinecone storage"""
    
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    
    def to_pinecone_format(self):
        """Convert to Pinecone upsert format"""
        return {
            "id": self.id,
            "values": self.embedding,
            "metadata": self.metadata,
            "sparse_values": {}  # For hybrid search if needed
        }


class PineconeMetadataSchema:
    """Define and validate Pinecone metadata schema"""
    
    # Metadata field types for filtering
    SCHEMA = {
        # Document info
        "doc_id": "string",
        "doc_title": "string",
        "doc_version": "string",
        
        # Hierarchy (enables structural queries)
        "h1": "string",  # Chapter
        "h2": "string",  # Section
        "h3": "string",  # Subsection
        
        # Location (enables source tracking)
        "page_number": "integer",
        "page_start": "integer",
        "page_end": "integer",
        "position_in_doc": "float",  # 0-1 range
        
        # Content classification (enables type-specific queries)
        "content_type": "string",  # [tutorial, reference, code, definition]
        "has_code": "boolean",
        "has_math": "boolean",
        "has_table": "boolean",
        
        # Technical level (enables audience filtering)
        "technical_level": "string",  # [beginner, intermediate, advanced]
        
        # Keywords (enables semantic tagging)
        "keywords": "string_array",
        
        # Chunking info
        "chunk_strategy": "string",
        "chunk_size": "integer",
        "chunk_overlap": "integer",
        "is_sub_split": "boolean",
        
        # Quality metrics
        "readability_score": "float",
        "chunk_completeness": "float",
        
        # Tracking
        "created_at": "string",
        "version": "string",
    }
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema"""
        for key, value in metadata.items():
            if key not in PineconeMetadataSchema.SCHEMA:
                print(f"⚠ Warning: Unknown metadata field '{key}'")
            
            # Type checking
            expected_type = PineconeMetadataSchema.SCHEMA.get(key)
            if expected_type == "integer" and not isinstance(value, int):
                return False
            if expected_type == "float" and not isinstance(value, (int, float)):
                return False
            if expected_type == "string" and not isinstance(value, str):
                return False
            if expected_type == "boolean" and not isinstance(value, bool):
                return False
            if expected_type == "string_array" and not isinstance(value, list):
                return False
        
        return True
    
    @staticmethod
    def print_schema():
        """Print schema for documentation"""
        print("\nPinecone Metadata Schema:")
        print("-" * 60)
        for field, field_type in PineconeMetadataSchema.SCHEMA.items():
            print(f"  {field:<25} {field_type}")


class PineconeQueryBuilder:
    """Build and execute Pinecone queries with metadata filtering"""
    
    @staticmethod
    def query_by_section(section: str, subsection: str = None) -> Dict:
        """Query by document section"""
        filter_spec = {"h2": section}
        if subsection:
            filter_spec["h3"] = subsection
        
        return filter_spec
    
    @staticmethod
    def query_by_chapter(chapter: str) -> Dict:
        """Query by chapter"""
        return {"h1": chapter}
    
    @staticmethod
    def query_by_content_type(content_type: str) -> Dict:
        """Query by content type"""
        return {"content_type": content_type}
    
    @staticmethod
    def query_code_only() -> Dict:
        """Query only code examples"""
        return {"has_code": True}
    
    @staticmethod
    def query_by_technical_level(level: str) -> Dict:
        """Query by technical level (beginner, intermediate, advanced)"""
        return {"technical_level": level}
    
    @staticmethod
    def query_by_page_range(start_page: int, end_page: int) -> Dict:
        """Query by page range"""
        return {
            "$and": [
                {"page_number": {"$gte": start_page}},
                {"page_number": {"$lte": end_page}}
            ]
        }
    
    @staticmethod
    def query_with_keywords(keywords: List[str]) -> Dict:
        """Query by keywords"""
        return {"keywords": {"$in": keywords}}
    
    @staticmethod
    def combine_filters(*filters: Dict) -> Dict:
        """Combine multiple filters with AND logic"""
        return {"$and": list(filters)}


class PineconeRAGIntegration:
    """High-level interface for RAG operations with Pinecone"""
    
    def __init__(self, index_name: str = "fintbx-rag", namespace: str = "production"):
        self.index_name = index_name
        self.namespace = namespace
        
        # Note: In production, initialize actual Pinecone client here
        # from pinecone import Pinecone
        # self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        # self.index = self.pc.Index(index_name)
        
        self.chunks_storage = {}  # For demo purposes
    
    def prepare_chunks_for_storage(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[PineconeChunk]:
        """
        Prepare chunks with embeddings for storage in Pinecone
        
        Args:
            chunks: List of chunk dicts with text and metadata
            embeddings: List of embedding vectors (same length as chunks)
        
        Returns:
            List of PineconeChunk objects ready for upsert
        """
        
        assert len(chunks) == len(embeddings), "Chunks and embeddings length mismatch"
        
        pinecone_chunks = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            
            # Validate metadata
            if not PineconeMetadataSchema.validate_metadata(chunk):
                print(f"⚠ Invalid metadata in chunk {i}")
            
            # Create Pinecone chunk
            pc_chunk = PineconeChunk(
                id=chunk.get("chunk_id", f"chunk_{i:06d}"),
                text=chunk.get("text", ""),
                embedding=embedding,
                metadata={
                    k: v for k, v in chunk.items()
                    if k not in ["text", "chunk_id"]
                }
            )
            
            pinecone_chunks.append(pc_chunk)
        
        return pinecone_chunks
    
    def format_for_upsert(self, pinecone_chunks: List[PineconeChunk]) -> List[Dict]:
        """Format chunks for Pinecone upsert operation"""
        return [chunk.to_pinecone_format() for chunk in pinecone_chunks]
    
    def store_chunks(self, pinecone_chunks: List[PineconeChunk], batch_size: int = 100):
        """
        Store chunks in Pinecone (with batching for efficiency)
        
        Args:
            pinecone_chunks: List of PineconeChunk objects
            batch_size: How many chunks to upsert at once
        """
        
        formatted_chunks = self.format_for_upsert(pinecone_chunks)
        
        print(f"\nStoring {len(formatted_chunks)} chunks to Pinecone...")
        print("-" * 60)
        
        # Process in batches
        for i in range(0, len(formatted_chunks), batch_size):
            batch = formatted_chunks[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(formatted_chunks) + batch_size - 1) // batch_size
            
            # In production: self.index.upsert(vectors=batch, namespace=self.namespace)
            # For demo:
            for chunk_dict in batch:
                self.chunks_storage[chunk_dict["id"]] = chunk_dict
            
            print(f"  Batch {batch_num}/{total_batches}: {len(batch)} chunks ✓")
        
        print(f"\n✓ Successfully stored {len(formatted_chunks)} chunks")
    
    def retrieve_with_metadata(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Dict = None
    ) -> List[Dict]:
        """
        Retrieve chunks with optional metadata filtering
        
        Args:
            query_vector: Embedding vector of the query
            top_k: Number of results to return
            metadata_filter: Pinecone filter expression
        
        Returns:
            List of retrieved chunks with metadata and text
        """
        
        # In production:
        # results = self.index.query(
        #     vector=query_vector,
        #     top_k=top_k,
        #     filter=metadata_filter,
        #     namespace=self.namespace,
        #     include_metadata=True,
        #     include_values=False
        # )
        
        # For demo - return mock results
        print(f"\nRetrieving top-{top_k} chunks...")
        if metadata_filter:
            print(f"Filter: {metadata_filter}")
        
        return []
    
    def search_by_chapter(self, query_vector: List[float], chapter: str, top_k: int = 5):
        """Search within a specific chapter"""
        return self.retrieve_with_metadata(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=PineconeQueryBuilder.query_by_chapter(chapter)
        )
    
    def search_by_content_type(
        self,
        query_vector: List[float],
        content_type: str,
        top_k: int = 5
    ):
        """Search for specific content type (tutorial, reference, code)"""
        return self.retrieve_with_metadata(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=PineconeQueryBuilder.query_by_content_type(content_type)
        )
    
    def search_code_examples(self, query_vector: List[float], top_k: int = 5):
        """Search for code examples only"""
        return self.retrieve_with_metadata(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=PineconeQueryBuilder.query_code_only()
        )
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored chunks"""
        
        stats = {
            "total_chunks": len(self.chunks_storage),
            "namespaces": [self.namespace],
            "index_name": self.index_name,
            
            # In production, get from Pinecone stats
            "metadata_fields": list(PineconeMetadataSchema.SCHEMA.keys()),
            "estimated_vectors": len(self.chunks_storage),
        }
        
        return stats


class RAGEvaluationMetrics:
    """Metrics for evaluating RAG performance with metadata-aware retrieval"""
    
    @staticmethod
    def calculate_mrr(retrieved_chunks: List[Dict], gold_chunk_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank
        Measures how early the first correct answer appears
        """
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.get("id") in gold_chunk_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def calculate_map_at_k(
        retrieved_chunks: List[Dict],
        gold_chunk_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Mean Average Precision @ k
        Average of precisions at each position where relevant chunk appears
        """
        precisions = []
        
        for i, chunk in enumerate(retrieved_chunks[:k]):
            if chunk.get("id") in gold_chunk_ids:
                relevant_so_far = sum(
                    1 for j in range(i+1)
                    if retrieved_chunks[j].get("id") in gold_chunk_ids
                )
                precisions.append(relevant_so_far / (i + 1))
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    @staticmethod
    def calculate_ncdg_at_k(
        retrieved_chunks: List[Dict],
        gold_chunk_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Normalized Cumulative Discounted Gain @ k
        Accounts for ranking quality, with diminishing returns for lower ranks
        """
        dcg = 0.0
        for i, chunk in enumerate(retrieved_chunks[:k]):
            relevance = 1.0 if chunk.get("id") in gold_chunk_ids else 0.0
            dcg += relevance / (1 + i)
        
        # Perfect ranking DCG
        ideal_dcg = sum(1.0 / (i + 1) for i in range(min(k, len(gold_chunk_ids))))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def calculate_section_coherence(retrieved_chunks: List[Dict]) -> float:
        """
        Are retrieved chunks from the same section? Higher = more coherent
        Helps measure if related content is retrieved together
        """
        if not retrieved_chunks:
            return 0.0
        
        sections = [c.get("h2", "unknown") for c in retrieved_chunks]
        unique_sections = len(set(sections))
        
        # All from same section = 1.0, all different = 0.0
        return 1.0 - (unique_sections - 1) / len(sections)


def print_evaluation_report(strategy_name: str, metrics: Dict):
    """Print formatted evaluation report"""
    print("\n" + "="*70)
    print(f"EVALUATION REPORT: {strategy_name}")
    print("="*70)
    print(f"MRR (Mean Reciprocal Rank): {metrics.get('mrr', 0):.3f}")
    print(f"MAP@5 (Mean Average Precision): {metrics.get('map_at_5', 0):.3f}")
    print(f"NDCG@5 (Normalized Discounted Cumulative Gain): {metrics.get('ndcg_at_5', 0):.3f}")
    print(f"Section Coherence: {metrics.get('coherence', 0):.3f}")
    print("="*70)


if __name__ == "__main__":
    # Demo: Show schema
    PineconeMetadataSchema.print_schema()
    
    # Demo: Show query examples
    print("\n\nQuery Examples:")
    print("-" * 60)
    
    queries = [
        ("Chapter Query", PineconeQueryBuilder.query_by_chapter("Chapter 1")),
        ("Content Type", PineconeQueryBuilder.query_by_content_type("tutorial")),
        ("Code Examples", PineconeQueryBuilder.query_code_only()),
        ("Technical Level", PineconeQueryBuilder.query_by_technical_level("beginner")),
    ]
    
    for query_name, query_spec in queries:
        print(f"\n{query_name}:")
        print(f"  {query_spec}")
    
    print("\n✓ Pinecone integration module ready for use")
