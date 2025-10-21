"""
Ingest chunks into Pinecone vector database
Loads chunks from experimental_framework output and uploads with embeddings
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings


class PineconeIngestion:
    """Handle ingestion of chunks into Pinecone"""
    
    def __init__(self):
        """Initialize Pinecone client and embeddings model"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "bigdata-assignment-03")
        self.embedding_model = os.getenv("PINECONE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimension = int(os.getenv("PINECONE_EMBEDDING_DIMENSION", "3072"))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get or create index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        print(f"✓ Connected to Pinecone index: {self.index_name}")
        print(f"✓ Using embedding model: {self.embedding_model} (dim: {self.embedding_dimension})")
    
    def load_chunks(self, chunk_file: Path) -> List[Dict[str, Any]]:
        """Load chunks from JSON file"""
        print(f"\n📂 Loading chunks from: {chunk_file}")
        
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✓ Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for all chunks"""
        print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            print(f"  ✓ Processed batch {(i//batch_size)+1}/{(len(texts)+batch_size-1)//batch_size}")
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def prepare_vectors(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[tuple]:
        """Prepare vectors for Pinecone upsert"""
        print(f"\n📦 Preparing vectors for Pinecone...")

        vectors = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validate embedding dimension first
            if len(embedding) != self.embedding_dimension:
                print(f"⚠ Warning: Embedding {i} has dimension {len(embedding)}, expected {self.embedding_dimension}")
                continue

            # Build base metadata from the chunk
            base_metadata = {
                k: v for k, v in chunk.items()
                if k not in ["text", "chunk_id", "embedding"] and v is not None
            }

            # Keep only primitive types (Pinecone requirement)
            metadata = {k: v for k, v in base_metadata.items() if isinstance(v, (str, int, float, bool))}

            # Add text (trim to keep metadata small)
            metadata["text"] = chunk.get("text", "")[:1000]

            # Create vector tuple: (id, embedding, metadata)
            vector = (
                chunk.get("chunk_id", f"chunk_{i:06d}"),
                embedding,
                metadata
            )
            vectors.append(vector)

        print(f"✓ Prepared {len(vectors)} vectors")
        return vectors
   
    def upsert_to_pinecone(self, vectors: List[tuple], batch_size: int = 100):
        """Upsert vectors to Pinecone"""
        print(f"\n⬆️  Upserting {len(vectors)} vectors to Pinecone...")
        
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Upsert batch
                self.index.upsert(vectors=batch, namespace="production")
                print(f"  ✓ Batch {batch_num}/{total_batches}: {len(batch)} vectors uploaded")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            except Exception as e:
                print(f"  ✗ Error upserting batch {batch_num}: {e}")
                raise
        
        print(f"✓ Successfully uploaded all vectors to Pinecone!")
    
    def verify_upload(self, num_samples: int = 5):
        """Verify that vectors were uploaded correctly"""
        print(f"\n✓ Verification: Checking index statistics...")
        
        try:
            stats = self.index.describe_index_stats()
            print(f"  Total vectors: {stats.total_vector_count}")
            print(f"  Namespaces: {list(stats.namespaces.keys())}")
            
            if stats.total_vector_count > 0:
                print(f"  ✓ Index is populated with {stats.total_vector_count} vectors!")
            else:
                print(f"  ⚠ Index appears to be empty")
        
        except Exception as e:
            print(f"  ⚠ Could not verify: {e}")
    
    def ingest(self, chunk_file: Path):
        """Main ingestion pipeline"""
        try:
            # Load chunks
            chunks = self.load_chunks(chunk_file)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare vectors
            vectors = self.prepare_vectors(chunks, embeddings)
            
            # Upsert to Pinecone
            self.upsert_to_pinecone(vectors)
            
            # Verify
            self.verify_upload()
            
            print("\n" + "="*70)
            print("✅ INGESTION COMPLETE!")
            print("="*70)
            print(f"Successfully ingested {len(chunks)} chunks into Pinecone")
            print(f"Index: {self.index_name}")
            print(f"Namespace: production")
            
        except Exception as e:
            print(f"\n❌ INGESTION FAILED: {e}")
            raise


def main():
    """Main ingestion run"""
    
    # Determine project root
    project_root = Path(__file__).parent.parent.parent
    
    # Use CodeAware strategy (optimal for MATLAB Financial Toolbox guide)
    # This strategy respects code block boundaries and is best for programming guides
    chunk_file = project_root / "data/rag_experiments/chunks_codeaware.json"
    
    # Alternative strategies:
    # chunk_file = project_root / "data/rag_experiments/chunks_hybrid.json"        # Good for structure-aware docs
    # chunk_file = project_root / "data/rag_experiments/chunks_markdownheader.json" # Good for narrative docs
    # chunk_file = project_root / "data/rag_experiments/chunks_recursive.json"      # Generic splitting
    
    print("="*70)
    print("PINECONE INGESTION - Upload Chunks with Embeddings")
    print("="*70)
    
    # Initialize ingestion
    ingestion = PineconeIngestion()
    
    # Run ingestion
    ingestion.ingest(chunk_file)


if __name__ == "__main__":
    main()
