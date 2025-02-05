import requests
from bs4 import BeautifulSoup
import os
from serpapi.google_search import GoogleSearch as search
import os
import aiohttp
from typing import Any, Dict, List, Optional

class ContentScraper:
    
    """
    A class to scrape web content, search Google for information, and retrieve stock price data using the SERP API.

    Attributes:
        serp_api_key (str): The API key to authenticate requests to the SERP API.
    """
    
    def __init__(self, serp_api_key):
        """
        Initializes the ContentScraper with a SERP API key.

        Args:
            serp_api_key (str): The API key used to authenticate requests to the SERP API.
        """
        self.serp_api_key = serp_api_key
        

    def scrape_content(self, url):
        """
        Scrapes and extracts the textual content from a webpage.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            str: The first 800 characters of the webpage's content if successful.
            None: If the request fails or the webpage content cannot be retrieved.
        """
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(paragraph.text for paragraph in paragraphs)
            return content[:800]
        except requests.RequestException:
            return None

    def search_google(self, query):
        """
        Searches Google for a given query using the SERP API and extracts relevant information.

        Args:
            query (str): The search query.

        Returns:
            tuple:
                - list: A list of dictionaries containing source URLs and descriptions.
                - list: A list of text snippets from the AI overview section of the search results.
        """
        
        params = {
            "engine": "google_finance",
            "q": query,
            "api_key": self.serp_api_key
        }

        store = search(params).get_dict()

        source_description_list = []

        if 'knowledge_graph' in store and store['knowledge_graph'] is not None:
            new_dict = {
                "source": store["knowledge_graph"].get("source", ""),
                "description": store["knowledge_graph"].get("description", "")
            }
            source_description_list.append(new_dict)

        if 'related_questions' in store and store['related_questions']:
            for question in store['related_questions']:
                source_description_list.append({
                    'source': question.get('link', ''),
                    'description': question.get('snippet', '')
                })

        ai_overview_context = []
        if 'ai_overview' in store and 'text_blocks' in store['ai_overview']:
          for block in store["ai_overview"]["text_blocks"]:
              # Check if 'snippet' is in the block
              if block.get("snippet"):
                  ai_overview_context.append(block["snippet"])

              # If 'list' is in the block, iterate through its items
              if block.get("list"):
                  for item in block["list"]:
                      if item.get("snippet"):
                          ai_overview_context.append(item["snippet"])

        return source_description_list, ai_overview_context

    def get_content_from_urls(self, source_description_list):
        """
        Fetches and compiles content from a list of URLs.

        Args:
            source_description_list (list): A list of dictionaries containing source URLs and their descriptions.

        Returns:
            tuple:
                - list: A list of dictionaries with URLs and their respective content.
                - list: A list of content strings from the URLs.
        """
        
        urls = [item["source"] for item in source_description_list]
        all_content = []
        context = []

        for url in urls:
            content = self.scrape_content(url)
            if content:
                all_content.append({"url": url, "content": content})
                context.append(content)

        return all_content, context
    
    def get_stock_price(self, query):
        """
        Gets stock price information if present in the store dictionary.
        Returns a list containing a formatted statement with stock price information.
        """
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serp_api_key
        }

        store = search(params).get_dict()
        stock_info = []
        if "answer_box" in store and store["answer_box"]:
            answer_box = store["answer_box"]
            if "list" in answer_box:
                stock_info.extend(answer_box["list"])
                

            # Ensure 'price' exists, then retrieve other optional fields if available
            if "price" in answer_box:
                price = answer_box["price"]
                stock = answer_box.get("stock", "Stock")
                currency = answer_box.get("currency", "")
                exchange = answer_box.get("exchange", "an Exchange")

                stock_info_more = [
                    f"According to {exchange}, the stock price is {currency} {price} for {stock}."
                ]
                stock_info.extend(stock_info_more)

        # Return empty list if stock price information is not found
        return stock_info




class GoogleSerperAPI:
    """
    A Python client for interacting with the Serper.dev API to perform Google searches 
    and retrieve search results.

    Attributes:
        api_key (str): The API key used to authenticate with the Serper.dev API.
        k (int): The number of search results to retrieve. Defaults to 10.
        gl (str): Geolocation of the search. Defaults to "us" (United States).
        hl (str): Language of the search results. Defaults to "en" (English).
        search_type (str): The type of search to perform (e.g., "search", "images"). Defaults to "search".
        initialised (bool): Indicates whether the instance is initialized with an API key.
    """
    
    def __init__(self, api_key: Optional[str] = None, k: int = 10, gl: str = "us", hl: str = "en", search_type: str = "search"):
        """
        Initializes the GoogleSerperAPI class with the provided API key and search configuration.

        Args:
            api_key (str, optional): API key for the Serper.dev API. If not provided, it reads from the environment variable `SERPER_API_KEY`.
            k (int, optional): Number of search results to retrieve. Defaults to 10.
            gl (str, optional): Geolocation for the search. Defaults to "us".
            hl (str, optional): Language for the search results. Defaults to "en".
            search_type (str, optional): Type of search (e.g., "search", "images"). Defaults to "search".

        Raises:
            ValueError: If the API key is not provided or available in the environment variables.
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Serper.dev is required.")
        self.k = k
        self.gl = gl
        self.hl = hl
        self.search_type = search_type
        self.initialised = True

    def _make_request(self, search_term: str, **kwargs: Any) -> Dict:
        """
        Makes a synchronous HTTP POST request to the Serper.dev API.

        Args:
            search_term (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            Dict: The JSON response from the Serper.dev API.

        Raises:
            requests.HTTPError: If the request fails.
        """
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{self.search_type}"
        params = {
            "q": search_term,
            "gl": self.gl,
            "hl": self.hl,
            "num": self.k,
            **kwargs,
        }
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()

    async def _make_async_request(self, search_term: str, **kwargs: Any) -> Dict:
        """
        Makes an asynchronous HTTP POST request to the Serper.dev API.

        Args:
            search_term (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            Dict: The JSON response from the Serper.dev API.

        Raises:
            aiohttp.ClientResponseError: If the request fails.
        """
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{self.search_type}"
        params = {
            "q": search_term,
            "gl": self.gl,
            "hl": self.hl,
            "num": self.k,
            **kwargs,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    def get_results(self, query: str, **kwargs: Any) -> Dict:
        """
        Retrieves search results for the given query.

        Args:
            query (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            Dict: The JSON response containing search results.
        """
        return self._make_request(query, **kwargs)

    async def get_async_results(self, query: str, **kwargs: Any) -> Dict:
        """
        Asynchronously retrieves search results for the given query.

        Args:
            query (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            Dict: The JSON response containing search results.
        """
        return await self._make_async_request(query, **kwargs)

    def parse_snippets(self, results: Dict) -> List[str]:
        """
        Extracts search result snippets from the JSON response.

        Args:
            results (Dict): The JSON response from the Serper.dev API.

        Returns:
            List[str]: A list of search result snippets or a default message if no snippets are found.
        """
        snippets = []
        if "organic" in results:
            for item in results["organic"][:self.k]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
        return snippets or ["No good results found."]

    def search(self, query: str, **kwargs: Any) -> str:
        """
        Performs a synchronous search and returns concatenated search result snippets.

        Args:
            query (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            str: A string of concatenated search result snippets.
        """
        results = self.get_results(query, **kwargs)
        return " ".join(self.parse_snippets(results))

    async def async_search(self, query: str, **kwargs: Any) -> str:
        """
        Performs an asynchronous search and returns concatenated search result snippets.

        Args:
            query (str): The search query.
            **kwargs: Additional parameters for the search request.

        Returns:
            str: A string of concatenated search result snippets.
        """
        results = await self.get_async_results(query, **kwargs)
        return " ".join(self.parse_snippets(results))
