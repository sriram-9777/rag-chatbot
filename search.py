# search.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

def google_search(query: str, num_results: int = 3) -> str:
    """
    Search Google and return top results as text
    """
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": SEARCH_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": num_results
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "items" not in data:
            return "No search results found."

        # Extract and combine search results
        results = []
        for item in data["items"]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append(f"Title: {title}\nSummary: {snippet}\nSource: {link}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Search failed: {str(e)}"