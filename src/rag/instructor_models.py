"""
Instructor Models for Structured Financial Concept Notes
Defines structured output schemas for financial concepts
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class Definition(BaseModel):
    """Detailed definition of a financial concept"""
    primary: str = Field(..., description="Primary definition of the concept")
    alternative: Optional[List[str]] = Field(None, description="Alternative definitions")
    context: str = Field(..., description="Context where this concept is used")
    code_examples: Optional[List[str]] = Field(None, description="Code snippets (MATLAB) demonstrating the concept")
    example_explanation: Optional[str] = Field(None, description="Detailed explanation of the code examples with context")


class KeyCharacteristics(BaseModel):
    """Key characteristics of the financial concept"""
    characteristics: List[str] = Field(..., description="List of key characteristics")
    importance: str = Field(..., description="Why this concept is important")


class Applications(BaseModel):
    """Practical applications of the financial concept"""
    use_cases: List[str] = Field(..., description="Real-world use cases")
    industry_examples: List[str] = Field(..., description="Examples from financial industry")
    matlab_relevance: Optional[str] = Field(None, description="How it relates to MATLAB Financial Toolbox")


class RelatedConcepts(BaseModel):
    """Related financial concepts"""
    related_terms: List[str] = Field(..., description="Related financial terms")
    relationships: List[str] = Field(..., description="How they are related")


class FinancialConceptNote(BaseModel):
    """Structured note for a financial concept"""
    concept_name: str = Field(..., description="Name of the financial concept")
    source: str = Field(..., description="Source of information (Wikipedia, etc.)")
    
    definition: Definition = Field(..., description="Detailed definition")
    characteristics: KeyCharacteristics = Field(..., description="Key characteristics")
    applications: Applications = Field(..., description="Practical applications")
    related_concepts: RelatedConcepts = Field(..., description="Related concepts")
    
    summary: str = Field(..., description="Brief summary (2-3 sentences)")
    key_takeaways: List[str] = Field(..., description="Key takeaways for practitioners")
    
    mathematical_foundation: Optional[str] = Field(None, description="Mathematical basis if applicable")
    risk_considerations: Optional[str] = Field(None, description="Associated risks")
    
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the extraction")


class FinancialTermCache(BaseModel):
    """Cache entry for a financial term"""
    term: str = Field(..., description="The financial term")
    wikipedia_url: str = Field(..., description="Wikipedia source URL")
    structured_note: FinancialConceptNote = Field(..., description="Structured note")
    cached_at: str = Field(..., description="Timestamp when cached")
    retrieved_from_cache: bool = Field(default=False, description="Whether this was from cache")


class QueryResponse(BaseModel):
    """Final response for a query about a financial concept"""
    query: str = Field(..., description="Original query")
    concept_found_in_vector: bool = Field(..., description="Was concept found in vector DB?")
    source: str = Field(..., description="Source: vector_db, wikipedia, or cache")
    
    structured_note: FinancialConceptNote = Field(..., description="Structured concept note")
    pinecone_context: Optional[str] = Field(None, description="Context from Pinecone if found")
    wikipedia_context: Optional[str] = Field(None, description="Context from Wikipedia")
    
    cached: bool = Field(default=False, description="Was this from cache?")
    processing_time_ms: float = Field(..., description="Time taken to process")


if __name__ == "__main__":
    # Example structured note
    example_note = FinancialConceptNote(
        concept_name="Volatility",
        source="Wikipedia",
        definition=Definition(
            primary="Volatility is a measure of the dispersion of returns for a security",
            alternative=["Standard deviation of returns", "Price fluctuation rate"],
            context="Used in options pricing and portfolio management"
        ),
        characteristics=KeyCharacteristics(
            characteristics=[
                "Indicates the degree of price variation",
                "Higher volatility means greater risk",
                "Can be calculated historically or implied"
            ],
            importance="Crucial for risk assessment and derivative pricing"
        ),
        applications=Applications(
            use_cases=[
                "Risk assessment in portfolios",
                "Options pricing (Black-Scholes)",
                "Value at Risk (VaR) calculations"
            ],
            industry_examples=[
                "Stock market volatility indices (VIX)",
                "Bond yield volatility",
                "Currency pair volatility"
            ],
            matlab_relevance="Calculated using MATLAB Financial Toolbox functions"
        ),
        related_concepts=RelatedConcepts(
            related_terms=["Standard Deviation", "Beta", "Value at Risk"],
            relationships=[
                "Volatility is the annualized standard deviation of returns",
                "Beta measures volatility relative to market",
                "VaR uses volatility in calculation"
            ]
        ),
        summary="Volatility measures the dispersion of returns and is a key metric in finance for assessing risk and pricing derivatives.",
        key_takeaways=[
            "Higher volatility indicates greater risk and price uncertainty",
            "Used extensively in options pricing models",
            "Can be calculated from historical data or implied from options prices"
        ],
        mathematical_foundation="Standard deviation of log returns, often annualized",
        risk_considerations="High volatility can lead to significant losses; important to monitor in portfolio management",
        confidence_score=0.95
    )
    
    print("Example Structured Note:")
    print(example_note.model_dump_json(indent=2))
