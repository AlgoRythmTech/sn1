"""
Web Search Utility for Supernova USLM - Add real-time web search capabilities
"""

import requests
import json
import logging
from typing import List, Dict, Optional, Any, Union
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serper API configuration
SERPER_API_KEY = "434dead0ffb79ac3df496b12254012837dd3ab1e"
SERPER_BASE_URL = "https://google.serper.dev/search"


class WebSearch:
    """Web search utility for Supernova USLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize web search with API key"""
        self.api_key = api_key or SERPER_API_KEY
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def search(
        self, 
        query: str, 
        num_results: int = 5,
        search_type: str = "search",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        safe_search: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a web search query
        
        Args:
            query: Search query string
            num_results: Number of results to return
            search_type: Type of search (search, images, news, places)
            include_domains: List of domains to include in results
            exclude_domains: List of domains to exclude from results
            safe_search: Whether to enable safe search filtering
            
        Returns:
            Dictionary containing search results
        """
        try:
            payload = {
                "q": query,
                "num": num_results,
                "gl": "us",  # Geographic location
                "hl": "en",  # Language
                "autocorrect": True,
                "type": search_type,
                "safe": safe_search
            }
            
            # Add domain filters if specified
            if include_domains:
                payload["includeDomains"] = include_domains
            if exclude_domains:
                payload["excludeDomains"] = exclude_domains
            
            logger.info(f"Searching web for: {query}")
            start_time = time.time()
            
            response = requests.post(
                SERPER_BASE_URL,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s")
            
            if response.status_code == 200:
                return self._process_results(response.json(), search_type)
            else:
                logger.error(f"Search failed with status code {response.status_code}: {response.text}")
                return {
                    "error": f"Search failed with status code {response.status_code}",
                    "status": "error",
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return {
                "error": str(e),
                "status": "error",
                "results": []
            }
    
    def _process_results(self, data: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Process and clean up search results"""
        processed_data = {
            "status": "success",
            "search_time": datetime.now().isoformat(),
            "query": data.get("searchParameters", {}).get("q", ""),
            "results": []
        }
        
        # Process organic search results
        if "organic" in data:
            for item in data["organic"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position"),
                    "source": item.get("source")
                }
                processed_data["results"].append(result)
        
        # Process knowledge graph if available
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            processed_data["knowledge_graph"] = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", ""),
                "attributes": kg.get("attributes", {}),
            }
        
        # Process answer box if available
        if "answerBox" in data:
            ab = data["answerBox"]
            processed_data["answer_box"] = {
                "title": ab.get("title", ""),
                "answer": ab.get("answer", ""),
                "snippet": ab.get("snippet", "")
            }
            
        return processed_data
    
    def format_results(self, results: Dict[str, Any], max_results: int = 3) -> str:
        """Format search results for human-readable output"""
        if results.get("status") == "error" or not results.get("results"):
            return "No relevant search results found."
        
        formatted = "Web Search Results:\n\n"
        
        # Add answer box if available (featured snippet)
        if "answer_box" in results:
            answer = results["answer_box"]
            if answer.get("answer"):
                formatted += f"Answer: {answer['answer']}\n\n"
            elif answer.get("snippet"):
                formatted += f"{answer['snippet']}\n\n"
        
        # Add knowledge graph if available
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            formatted += f"{kg['title']} ({kg.get('type', '')}): {kg.get('description', '')}\n\n"
        
        # Add organic results
        result_count = min(max_results, len(results["results"]))
        for i in range(result_count):
            result = results["results"][i]
            formatted += f"{i+1}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   Source: {result['source']}\n\n"
        
        return formatted
    
    def search_and_format(
        self, 
        query: str, 
        num_results: int = 5,
        max_display: int = 3
    ) -> str:
        """Search and return formatted results in one step"""
        results = self.search(query, num_results=num_results)
        return self.format_results(results, max_results=max_display)


class SearchEnhancedResponse:
    """Generate responses enhanced with web search results"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.search = WebSearch(api_key)
    
    def enhance_response(
        self, 
        query: str, 
        response: str, 
        should_search: bool = True,
        num_results: int = 5
    ) -> str:
        """
        Enhance an AI response with web search results when appropriate
        
        Args:
            query: User query
            response: Original AI response
            should_search: Whether web search should be performed
            num_results: Number of search results to include
            
        Returns:
            Enhanced response with search results if applicable
        """
        if not should_search:
            return response
            
        # Determine if query needs factual information
        needs_facts = self._query_needs_facts(query)
        
        if needs_facts:
            # Get search results
            search_results = self.search.search(query, num_results=num_results)
            
            if search_results.get("status") == "success" and search_results.get("results"):
                # Format search results
                formatted_results = self.search.format_results(
                    search_results,
                    max_results=min(3, num_results)
                )
                
                # Combine original response with search results
                enhanced = f"{response}\n\n---\n\n{formatted_results}"
                return enhanced
                
        return response
    
    def _query_needs_facts(self, query: str) -> bool:
        """Determine if a query likely needs factual information"""
        factual_indicators = [
            "who", "what", "when", "where", "why", "how",
            "explain", "tell me about", "information on",
            "latest", "current", "recent", "news", "data",
            "statistics", "facts", "history", "definition",
            "research", "studies", "report", "example"
        ]
        
        query_lower = query.lower()
        
        # Check for factual indicators
        for indicator in factual_indicators:
            if indicator in query_lower:
                return True
                
        return False


def search_query(query: str, api_key: Optional[str] = None) -> str:
    """Utility function to search and format results"""
    search = WebSearch(api_key)
    results = search.search(query)
    return search.format_results(results)


if __name__ == "__main__":
    # Test web search
    search = WebSearch()
    query = "What is AlgoRythm Tech?"
    results = search.search(query)
    print(search.format_results(results))
    
    # Test enhanced response
    enhancer = SearchEnhancedResponse()
    original_response = "AlgoRythm Tech is an AI company."
    enhanced = enhancer.enhance_response(query, original_response)
    print("\nEnhanced response:")
    print(enhanced)
