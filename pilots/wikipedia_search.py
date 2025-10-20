"""
Wikipedia Search Script
Searches Wikipedia for queries and returns relevant results
"""

import wikipedia
from typing import Optional, List, Dict


def search_wikipedia(query: str, results_count: int = 5) -> Dict:
    """
    Search Wikipedia for a query
    
    Args:
        query: Search query string
        results_count: Number of results to return (default: 5)
    
    Returns:
        Dictionary with search results and summary
    """
    try:
        # Search for the query
        search_results = wikipedia.search(query, results=results_count)
        
        if not search_results:
            return {
                "success": False,
                "query": query,
                "message": "No results found",
                "results": []
            }
        
        results = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary[:300] + "..." if len(page.summary) > 300 else page.summary,
                    "full_summary": page.summary,
                    "categories": page.categories[:5] if page.categories else [],
                    "links": page.links[:5] if page.links else []
                })
            except wikipedia.exceptions.DisambiguationError as e:
                results.append({
                    "title": title,
                    "error": "Disambiguation page",
                    "options": e.options[:5]
                })
            except wikipedia.exceptions.PageError:
                continue
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "results": []
        }


def get_wikipedia_summary(title: str) -> Dict:
    """
    Get full summary of a Wikipedia page by title
    
    Args:
        title: Wikipedia page title
    
    Returns:
        Dictionary with page content
    """
    try:
        page = wikipedia.page(title, auto_suggest=True)
        return {
            "success": True,
            "title": page.title,
            "url": page.url,
            "summary": page.summary,
            "content": page.content,
            "categories": page.categories,
            "links": page.links[:20]
        }
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "success": False,
            "error": "Disambiguation page",
            "options": e.options
        }
    except wikipedia.exceptions.PageError:
        return {
            "success": False,
            "error": f"Page '{title}' not found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def search_and_summarize(query: str) -> Dict:
    """
    Search Wikipedia and return the best matching page summary
    
    Args:
        query: Search query string
    
    Returns:
        Dictionary with top result summary
    """
    search_results = search_wikipedia(query, results_count=1)
    
    if search_results["success"] and search_results["results"]:
        top_result = search_results["results"][0]
        detailed_page = get_wikipedia_summary(top_result["title"])
        return detailed_page
    
    return search_results


def interactive_search():
    """
    Interactive Wikipedia search tool
    Allows user to search and explore Wikipedia
    """
    print("\n" + "="*70)
    print("Wikipedia Search Tool")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("1) Search Wikipedia")
        print("2) Get page summary")
        print("3) Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                results = search_wikipedia(query)
                
                if results["success"]:
                    print(f"\nFound {results['results_count']} results for '{results['query']}':\n")
                    for i, result in enumerate(results["results"], 1):
                        print(f"{i}. {result.get('title', 'N/A')}")
                        print(f"   URL: {result.get('url', 'N/A')}")
                        print(f"   Summary: {result.get('summary', 'N/A')}\n")
                else:
                    print(f"Error: {results.get('message', 'Unknown error')}")
        
        elif choice == "2":
            title = input("Enter Wikipedia page title: ").strip()
            if title:
                result = get_wikipedia_summary(title)
                
                if result["success"]:
                    print(f"\n{'='*70}")
                    print(f"Title: {result['title']}")
                    print(f"URL: {result['url']}")
                    print(f"{'='*70}")
                    print(f"\nSummary:\n{result['summary']}\n")
                    print(f"Content Length: {len(result['content'])} characters")
                    if result["categories"]:
                        print(f"Categories: {', '.join(result['categories'][:5])}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import json
    
    # Example usage
    # print("Example 1: Search for 'Python programming'")
    # results = search_wikipedia("Python programming", results_count=3)
    # print(json.dumps(results, indent=2))
    
    # print("\n" + "="*70)
    # print("Example 2: Get detailed summary of 'Machine Learning'")
    # detailed = get_wikipedia_summary("Machine Learning")
    # if detailed["success"]:
    #     print(f"Title: {detailed['title']}")
    #     print(f"Summary: {detailed['summary'][:500]}...\n")
    
    # print("\n" + "="*70)
    # print("Example 3: Search and summarize 'Artificial Intelligence'")
    # summary = search_and_summarize("Artificial Intelligence")
    # if summary["success"]:
    #     print(f"Title: {summary['title']}")
    #     print(f"Summary: {summary['summary'][:500]}...\n")
    
    print("\n" + "="*70)
    print("Starting interactive mode...\n")
    # Uncomment to start interactive search
    interactive_search()
