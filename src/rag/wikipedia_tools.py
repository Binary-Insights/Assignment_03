"""
LangChain Tools for Wikipedia Search
Integrates Wikipedia as a fallback source for financial concepts
"""

import wikipedia
import logging
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WikipediaSearchInput(BaseModel):
    """Input for Wikipedia search tool"""
    query: str = Field(..., description="The financial concept to search for on Wikipedia")
    num_results: int = Field(default=1, description="Number of results to return")


@tool("search_wikipedia", args_schema=WikipediaSearchInput)
def search_wikipedia_for_concept(query: str, num_results: int = 1) -> Dict[str, Any]:
    """
    Search Wikipedia for a financial concept.
    This tool is called when a financial concept is not found in the vector database.
    
    Args:
        query: The financial concept to search for
        num_results: Number of results to return
    
    Returns:
        Dictionary with Wikipedia search results
    """
    try:
        logger.info(f"Searching Wikipedia for: {query}")
        
        # Search Wikipedia
        search_results = wikipedia.search(query, results=num_results)
        
        if not search_results:
            logger.warning(f"No Wikipedia results found for: {query}")
            return {
                "success": False,
                "query": query,
                "message": "No Wikipedia results found",
                "results": []
            }
        
        results = []
        for title in search_results[:num_results]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                result = {
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary,
                    "content": page.content[:2000],  # Truncate for efficiency
                    "categories": page.categories[:5] if page.categories else []
                }
                results.append(result)
                logger.info(f"Retrieved Wikipedia page: {page.title}")
            
            except wikipedia.exceptions.DisambiguationError as e:
                logger.warning(f"Disambiguation page for: {title}")
                # Try to get the first option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        result = {
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary,
                            "content": page.content[:2000],
                            "categories": page.categories[:5] if page.categories else []
                        }
                        results.append(result)
                    except:
                        pass
            
            except wikipedia.exceptions.PageError:
                logger.warning(f"Page not found: {title}")
                continue
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {e}")
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "results": []
        }


@tool("get_wikipedia_full_content", args_schema=WikipediaSearchInput)
def _get_wikipedia_full_content_tool(query: str, num_results: int = 1) -> Dict[str, Any]:
    """
    Get full Wikipedia content for a financial concept (LangChain tool version).
    
    Args:
        query: The financial concept to search for
        num_results: Number of results to return
    
    Returns:
        Dictionary with full Wikipedia content
    """
    return get_wikipedia_full_content_impl(query, num_results)


def get_wikipedia_full_content(query: str, num_results: int = 1) -> Dict[str, Any]:
    """
    Get full Wikipedia content for a financial concept (direct function).
    
    Args:
        query: The financial concept to search for
        num_results: Number of results to return
    
    Returns:
        Dictionary with full Wikipedia content
    """
    return get_wikipedia_full_content_impl(query, num_results)


def get_wikipedia_full_content_impl(query: str, num_results: int = 1) -> Dict[str, Any]:
    """
    Get full Wikipedia content for a financial concept.
    
    Args:
        query: The financial concept to search for
        num_results: Number of results to return
    
    Returns:
        Dictionary with full Wikipedia content
    """
    try:
        logger.info(f"Getting full Wikipedia content for: {query}")
        
        search_results = wikipedia.search(query, results=num_results)
        
        if not search_results:
            return {
                "success": False,
                "query": query,
                "message": "No Wikipedia results found"
            }
        
        # Get the first result
        try:
            page = wikipedia.page(search_results[0], auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                page = wikipedia.page(e.options[0], auto_suggest=False)
            else:
                return {
                    "success": False,
                    "query": query,
                    "message": "Disambiguation page with no clear options"
                }
        except wikipedia.exceptions.PageError:
            return {
                "success": False,
                "query": query,
                "message": f"Page not found for query: {query}"
            }
        
        return {
            "success": True,
            "query": query,
            "title": page.title,
            "url": page.url,
            "content": page.content,
            "summary": page.summary,
            "categories": page.categories if page.categories else [],
            "links": page.links[:20] if page.links else []
        }
    
    except Exception as e:
        logger.error(f"Error getting Wikipedia content: {e}")
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }


def get_tools():
    """Return list of Wikipedia search tools for LangChain"""
    return [search_wikipedia_for_concept, _get_wikipedia_full_content_tool]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the tools
    print("Testing search_wikipedia_for_concept tool:")
    result = search_wikipedia_for_concept("Volatility Finance")
    print(result)
    
    print("\n" + "="*70 + "\n")
    
    print("Testing get_wikipedia_full_content tool:")
    result = get_wikipedia_full_content("Option Pricing")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content length: {len(result['content'])} characters")
