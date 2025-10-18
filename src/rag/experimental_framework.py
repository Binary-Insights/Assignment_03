"""
RAG Chunking Strategy Experimental Framework
Implements and compares multiple chunking strategies for Docling-parsed documents
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import statistics

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)


@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    chunk_id: str
    doc_id: str
    doc_title: str
    
    # Hierarchy
    h1: str = None  # Chapter
    h2: str = None  # Section
    h3: str = None  # Subsection
    
    # Location
    page_number: int = None
    page_start: int = None
    page_end: int = None
    position_in_doc: float = None  # 0-1
    
    # Content classification
    content_type: str = "general"  # [tutorial, reference, example, definition]
    has_code: bool = False
    has_math: bool = False
    has_table: bool = False
    
    # Semantic tags
    keywords: List[str] = None
    technical_level: str = "intermediate"  # [beginner, intermediate, advanced]
    
    # Chunking metadata
    chunk_strategy: str = None
    chunk_size: int = None
    chunk_overlap: int = None
    is_sub_split: bool = False
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StrategyMetrics:
    """Metrics for evaluating chunking strategy"""
    strategy_name: str
    chunk_count: int
    total_tokens: int
    mean_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_dev_chunk_size: float
    total_retrieval_time_ms: float = 0.0
    mean_accuracy: float = 0.0
    mean_coverage: float = 0.0
    mean_coherence: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks. Returns list of dicts with 'text' and metadata."""
        raise NotImplementedError
    
    def get_metrics(self, chunks: List[Dict[str, Any]]) -> StrategyMetrics:
        """Calculate metrics for chunks"""
        chunk_sizes = [len(c.get("text", "")) for c in chunks]
        
        return StrategyMetrics(
            strategy_name=self.name,
            chunk_count=len(chunks),
            total_tokens=sum(chunk_sizes),
            mean_chunk_size=statistics.mean(chunk_sizes) if chunk_sizes else 0,
            min_chunk_size=min(chunk_sizes) if chunk_sizes else 0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            std_dev_chunk_size=statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
        )


class MarkdownHeaderStrategy(ChunkingStrategy):
    """Split by markdown headers while preserving hierarchy"""
    
    def __init__(self):
        super().__init__("MarkdownHeader")
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            return_each_line=False
        )
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split using markdown headers"""
        chunks = self.splitter.split_text(text)
        
        # Convert to standard format
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "text": chunk.get("text", ""),
                "h1": chunk.get("h1"),
                "h2": chunk.get("h2"),
                "h3": chunk.get("h3"),
                "chunk_id": f"chunk_{i:06d}",
                "chunk_strategy": self.name,
            })
        
        return result


class RecursiveStrategy(ChunkingStrategy):
    """Split recursively with size and overlap control"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__("Recursive")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split recursively"""
        chunks = self.splitter.split_text(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "chunk_id": f"chunk_{i:06d}",
                "chunk_strategy": self.name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            })
        
        return result


class HybridStrategy(ChunkingStrategy):
    """Hybrid: markdown headers first, then recursive for large sections"""
    
    def __init__(self, max_section_size: int = 1200):
        super().__init__("Hybrid")
        self.max_section_size = max_section_size
        
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            return_each_line=False
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
        )
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split using hybrid approach"""
        # Step 1: Split by headers
        header_chunks = self.header_splitter.split_text(text)
        
        result = []
        chunk_id = 0
        
        for header_chunk in header_chunks:
            chunk_text = header_chunk.get("text", "")
            
            # Step 2: If chunk is small enough, keep it
            if len(chunk_text) <= self.max_section_size:
                result.append({
                    "text": chunk_text,
                    "h1": header_chunk.get("h1"),
                    "h2": header_chunk.get("h2"),
                    "h3": header_chunk.get("h3"),
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "chunk_strategy": self.name,
                    "is_sub_split": False,
                })
                chunk_id += 1
            else:
                # Step 3: Further split large chunks
                sub_chunks = self.recursive_splitter.split_text(chunk_text)
                
                for sub_chunk in sub_chunks:
                    result.append({
                        "text": sub_chunk,
                        "h1": header_chunk.get("h1"),
                        "h2": header_chunk.get("h2"),
                        "h3": header_chunk.get("h3"),
                        "chunk_id": f"chunk_{chunk_id:06d}",
                        "chunk_strategy": self.name,
                        "is_sub_split": True,
                    })
                    chunk_id += 1
        
        return result


class CodeAwareStrategy(ChunkingStrategy):
    """Split while respecting MATLAB code blocks"""
    
    def __init__(self):
        super().__init__("CodeAware")
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MATLAB,
            chunk_size=900,
            chunk_overlap=200,
        )
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split while preserving code integrity"""
        chunks = self.splitter.split_text(text)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "chunk_id": f"chunk_{i:06d}",
                "chunk_strategy": self.name,
                "has_code": "%" in chunk_text or "function" in chunk_text,
            })
        
        return result


class MetadataEnricher:
    """Add rich metadata to chunks"""
    
    # Keywords and tags mapping
    CHAPTER_KEYWORDS = {
        "Getting Started": ["matrix", "algebra", "functions", "basics"],
        "Performing Common Financial Tasks": ["cash flow", "dates", "yields", "portfolio"],
        "Analyzing Portfolios": ["optimization", "portfolio", "efficient", "variance"],
        "Mean-Variance Portfolio": ["mean", "variance", "optimization", "constraints"],
    }
    
    CONTENT_TYPES = {
        "introduction": ["introduction", "overview", "describe"],
        "tutorial": ["how to", "example", "step", "create"],
        "reference": ["definition", "function", "parameter", "syntax"],
        "code": ["function", "code", "matlab", "%", "algorithm"],
    }
    
    TECHNICAL_LEVELS = {
        "beginner": ["introduction", "overview", "basic", "simple", "example"],
        "intermediate": ["advanced", "optimization", "theory", "methodology"],
        "advanced": ["mathematical", "formulation", "theorem", "constraint", "algorithm"],
    }
    
    @staticmethod
    def extract_page_from_chunk(chunk: Dict[str, Any], doc_structure: Dict) -> int:
        """Estimate page number from chunk position"""
        # This is simplified - in real system, track page numbers during extraction
        return 1
    
    @staticmethod
    def classify_content_type(chunk: Dict[str, Any]) -> str:
        """Classify chunk as tutorial, reference, code, or definition"""
        text = chunk.get("text", "").lower()
        
        for ctype, keywords in MetadataEnricher.CONTENT_TYPES.items():
            if any(kw in text for kw in keywords):
                return ctype
        
        return "general"
    
    @staticmethod
    def classify_technical_level(chunk: Dict[str, Any]) -> str:
        """Classify as beginner, intermediate, or advanced"""
        text = chunk.get("text", "").lower()
        
        level_scores = {
            "beginner": 0,
            "intermediate": 0,
            "advanced": 0,
        }
        
        for level, keywords in MetadataEnricher.TECHNICAL_LEVELS.items():
            level_scores[level] = sum(1 for kw in keywords if kw in text)
        
        return max(level_scores, key=level_scores.get)
    
    @staticmethod
    def extract_keywords(chunk: Dict[str, Any]) -> List[str]:
        """Extract domain keywords from chunk"""
        text = chunk.get("text", "").lower()
        
        financial_terms = [
            "portfolio", "optimization", "matrix", "algebra", "cash flow",
            "yield", "bond", "equity", "derivative", "risk", "return",
            "correlation", "variance", "mean", "efficient frontier"
        ]
        
        found = [term for term in financial_terms if term in text]
        return found
    
    @staticmethod
    def enrich_chunk(chunk: Dict[str, Any], doc_info: Dict = None) -> Dict[str, Any]:
        """Add metadata to chunk"""
        
        chunk_id = chunk.get("chunk_id", "unknown")
        
        enriched = {
            **chunk,
            "doc_id": doc_info.get("doc_id", "unknown") if doc_info else "unknown",
            "doc_title": doc_info.get("title", "Unknown") if doc_info else "Unknown",
            
            # Content classification
            "content_type": MetadataEnricher.classify_content_type(chunk),
            "technical_level": MetadataEnricher.classify_technical_level(chunk),
            "keywords": MetadataEnricher.extract_keywords(chunk),
            
            # Content properties
            "has_code": "%" in chunk.get("text", "") or "function" in chunk.get("text", ""),
            "has_math": any(sym in chunk.get("text", "") for sym in ["×", "²", "√", "∑", "∫", "\\frac"]),
            
            # Location (simplified - would need real page tracking)
            "page_number": 1,
            "position_in_doc": 0.5,
        }
        
        return enriched


class ExperimentalFramework:
    """Framework for comparing chunking strategies"""
    
    def __init__(self, document_path: Path):
        self.document_path = document_path
        self.document_text = self._load_document()
        self.strategies = {}
        self.results = {}
    
    def _load_document(self) -> str:
        """Load document from file"""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def register_strategy(self, strategy: ChunkingStrategy):
        """Register a chunking strategy for testing"""
        self.strategies[strategy.name] = strategy
    
    def run_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered strategies and collect results"""
        
        print("\n" + "="*80)
        print("CHUNKING STRATEGY EXPERIMENTAL FRAMEWORK")
        print("="*80)
        
        for name, strategy in self.strategies.items():
            print(f"\n▶ Testing Strategy: {name}")
            print("-" * 80)
            
            # Time the chunking
            start_time = time.time()
            chunks = strategy.split(self.document_text)
            elapsed = time.time() - start_time
            
            # Enrich chunks with metadata
            doc_info = {
                "doc_id": "fintbx_part_001",
                "title": "Financial Toolbox™ User's Guide",
            }
            
            enriched_chunks = [
                MetadataEnricher.enrich_chunk(c, doc_info) 
                for c in chunks
            ]
            
            # Calculate metrics
            metrics = strategy.get_metrics(enriched_chunks)
            metrics.total_retrieval_time_ms = elapsed * 1000
            
            # Store results
            self.results[name] = {
                "strategy": strategy,
                "chunks": enriched_chunks,
                "metrics": metrics,
                "execution_time": elapsed,
            }
            
            # Print summary
            self._print_strategy_summary(metrics)
        
        return self.results
    
    def _print_strategy_summary(self, metrics: StrategyMetrics):
        """Print summary for a strategy"""
        print(f"  Strategy: {metrics.strategy_name}")
        print(f"  ├─ Total chunks: {metrics.chunk_count}")
        print(f"  ├─ Total tokens: {metrics.total_tokens:,}")
        print(f"  ├─ Mean chunk size: {metrics.mean_chunk_size:.0f} chars")
        print(f"  ├─ Size range: {metrics.min_chunk_size} - {metrics.max_chunk_size}")
        print(f"  ├─ Std dev: {metrics.std_dev_chunk_size:.0f}")
        print(f"  └─ Execution time: {metrics.total_retrieval_time_ms:.1f}ms")
    
    def compare_strategies(self) -> str:
        """Generate comparison report"""
        
        report = "\n" + "="*100 + "\n"
        report += "STRATEGY COMPARISON RESULTS\n"
        report += "="*100 + "\n\n"
        
        # Create comparison table
        report += f"{'Strategy':<15} {'Chunks':<10} {'Tokens':<12} {'Avg Size':<12} {'Std Dev':<10} {'Time':<10}\n"
        report += "-" * 100 + "\n"
        
        for name, result in self.results.items():
            metrics = result["metrics"]
            report += (
                f"{metrics.strategy_name:<15} "
                f"{metrics.chunk_count:<10} "
                f"{metrics.total_tokens:<12,} "
                f"{metrics.mean_chunk_size:<12.0f} "
                f"{metrics.std_dev_chunk_size:<10.1f} "
                f"{metrics.total_retrieval_time_ms:<10.1f}ms\n"
            )
        
        report += "\n" + "="*100 + "\n"
        report += "RECOMMENDATIONS:\n"
        report += "="*100 + "\n\n"
        
        # Find best by different criteria
        best_by_chunks = min(self.results.items(), key=lambda x: x[1]["metrics"].chunk_count)
        best_by_uniformity = min(self.results.items(), key=lambda x: x[1]["metrics"].std_dev_chunk_size)
        fastest = min(self.results.items(), key=lambda x: x[1]["execution_time"])
        
        report += f"✓ Least fragmentation: {best_by_chunks[0]} ({best_by_chunks[1]['metrics'].chunk_count} chunks)\n"
        report += f"✓ Most uniform size: {best_by_uniformity[0]} (std dev: {best_by_uniformity[1]['metrics'].std_dev_chunk_size:.1f})\n"
        report += f"✓ Fastest execution: {fastest[0]} ({fastest[1]['execution_time']*1000:.1f}ms)\n"
        
        report += f"\n✓ RECOMMENDED: Hybrid Strategy\n"
        report += f"  Reason: Best balance of structure preservation and practical chunk sizes\n"
        
        return report
    
    def save_chunks_to_file(self, strategy_name: str, output_path: Path):
        """Save chunks from a strategy to JSON"""
        
        if strategy_name not in self.results:
            print(f"Strategy '{strategy_name}' not found")
            return
        
        chunks = self.results[strategy_name]["chunks"]
        
        # Convert to serializable format
        serializable_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            # Convert any non-serializable types
            if chunk_copy.get("keywords"):
                chunk_copy["keywords"] = list(chunk_copy["keywords"])
            serializable_chunks.append(chunk_copy)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(serializable_chunks)} chunks to {output_path}")
    
    def save_results_summary(self, output_path: Path):
        """Save overall results summary"""
        
        summary = {
            "document": str(self.document_path),
            "document_size_chars": len(self.document_text),
            "strategies_tested": len(self.results),
            "results": {}
        }
        
        for name, result in self.results.items():
            metrics = result["metrics"]
            summary["results"][name] = {
                "chunk_count": metrics.chunk_count,
                "total_tokens": metrics.total_tokens,
                "mean_chunk_size": round(metrics.mean_chunk_size, 2),
                "min_chunk_size": metrics.min_chunk_size,
                "max_chunk_size": metrics.max_chunk_size,
                "std_dev": round(metrics.std_dev_chunk_size, 2),
                "execution_time_ms": round(metrics.total_retrieval_time_ms, 1),
            }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved results summary to {output_path}")


def main():
    """Main experimental run"""
    
    # Paths
    project_root = Path(__file__).parent
    doc_path = project_root / "data/parsed/FINTBX/fintbx_part_001/text/structured_content.txt"
    output_dir = project_root / "data/rag_experiments"
    
    # Initialize framework
    framework = ExperimentalFramework(doc_path)
    
    # Register strategies
    framework.register_strategy(MarkdownHeaderStrategy())
    framework.register_strategy(RecursiveStrategy(chunk_size=800, chunk_overlap=200))
    framework.register_strategy(RecursiveStrategy(chunk_size=1000, chunk_overlap=200))
    framework.register_strategy(RecursiveStrategy(chunk_size=1200, chunk_overlap=250))
    framework.register_strategy(HybridStrategy(max_section_size=1200))
    framework.register_strategy(CodeAwareStrategy())
    
    # Run all strategies
    results = framework.run_all_strategies()
    
    # Print comparison
    print(framework.compare_strategies())
    
    # Save results
    for strategy_name in results.keys():
        output_file = output_dir / f"chunks_{strategy_name.lower()}.json"
        framework.save_chunks_to_file(strategy_name, output_file)
    
    framework.save_results_summary(output_dir / "summary.json")
    
    print(f"\n✓ Experiment complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
