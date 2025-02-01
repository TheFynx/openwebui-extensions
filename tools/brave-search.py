"""
title: Brave Search Tool
description: This tool performs web and local searches using the Brave Search API to get real-time information from the internet
required_open_webui_version: 0.4.0
requirements: aiohttp
version: 1.0.0
licence: MIT
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import aiohttp
import asyncio
from datetime import datetime

class Tools:
    def __init__(self):
        """Initialize the Tool with valves."""
        self.valves = self.Valves()
        self._last_request_time = 0
        self._monthly_requests = 0
        self._last_month = datetime.now().month

    class Valves(BaseModel):
        model_config = {
            "arbitrary_types_allowed": True
        }

        brave_api_key: str = Field(
            "",
            description="Your Brave Search API key"
        )
        max_results: int = Field(
            5,
            description="Maximum number of search results to return (1-20)"
        )

    async def _check_rate_limit(self):
        """
        Implement rate limiting (1 request per second, 15000 per month).
        Raises RuntimeError if rate limit is exceeded.
        """
        current_time = datetime.now()
        current_month = current_time.month

        # Reset monthly counter if month changed
        if current_month != self._last_month:
            self._monthly_requests = 0
            self._last_month = current_month

        # Check monthly limit
        if self._monthly_requests >= 15000:
            raise RuntimeError("Monthly rate limit (15000) exceeded")

        # Ensure 1 second between requests
        time_since_last = (current_time.timestamp() - self._last_request_time)
        if time_since_last < 1:
            await asyncio.sleep(1 - time_since_last)

        self._last_request_time = current_time.timestamp()
        self._monthly_requests += 1

    async def _perform_web_search(self, query: str, count: int) -> str:
        """Perform a web search using Brave Search API."""
        url = 'https://api.search.brave.com/res/v1/web/search'
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.valves.brave_api_key
        }
        params = {
            'q': query,
            'count': min(count, 20)
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Brave API error: {response.status} {error_text}")

                data = await response.json()
                results = data.get('web', {}).get('results', [])

                if not results:
                    return "No results found for the given query."

                formatted_results = "Web Search Results:\n\n"
                for i, result in enumerate(results, 1):
                    formatted_results += f"{i}. {result.get('title', 'No title')}\n"
                    formatted_results += f"URL: {result.get('url', 'No URL')}\n"
                    if description := result.get('description'):
                        formatted_results += f"Description: {description}\n"
                    formatted_results += "\n"

                return formatted_results

    async def _get_location_details(self, location_ids: List[str]) -> tuple:
        """Get detailed information about locations."""
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.valves.brave_api_key
        }

        async with aiohttp.ClientSession() as session:
            # Get POIs data
            poi_url = 'https://api.search.brave.com/res/v1/local/pois'
            params = [('ids', id) for id in location_ids]
            async with session.get(poi_url, headers=headers, params=params) as response:
                if response.status != 200:
                    raise RuntimeError(f"Brave API error: {response.status}")
                pois_data = await response.json()

            # Get descriptions
            desc_url = 'https://api.search.brave.com/res/v1/local/descriptions'
            async with session.get(desc_url, headers=headers, params=params) as response:
                if response.status != 200:
                    raise RuntimeError(f"Brave API error: {response.status}")
                desc_data = await response.json()

        return pois_data, desc_data

    async def _perform_local_search(self, query: str, count: int) -> str:
        """Perform a local search using Brave Search API."""
        url = 'https://api.search.brave.com/res/v1/web/search'
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.valves.brave_api_key
        }
        params = {
            'q': query,
            'count': min(count, 20),
            'search_lang': 'en',
            'result_filter': 'locations'
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Brave API error: {response.status} {error_text}")

                data = await response.json()
                location_results = data.get('locations', {}).get('results', [])

                if not location_results:
                    # Fallback to web search if no local results
                    return await self._perform_web_search(query, count)

                location_ids = [r['id'] for r in location_results if 'id' in r]
                if not location_ids:
                    return "No local results found for the given query."

                # Get detailed information about locations
                await self._check_rate_limit()  # Additional rate limit check for the second request
                pois_data, desc_data = await self._get_location_details(location_ids)

            formatted_results = "Local Search Results:\n\n"
            for poi in pois_data.get('results', []):
                name = poi.get('name', 'No name available')
                formatted_results += f"Name: {name}\n"

                # Format address
                address_parts = []
                if address := poi.get('address', {}):
                    if street := address.get('streetAddress'):
                        address_parts.append(street)
                    if locality := address.get('addressLocality'):
                        address_parts.append(locality)
                    if region := address.get('addressRegion'):
                        address_parts.append(region)
                    if postal := address.get('postalCode'):
                        address_parts.append(postal)
                formatted_results += f"Address: {', '.join(address_parts) if address_parts else 'N/A'}\n"

                # Add phone
                if phone := poi.get('phone'):
                    formatted_results += f"Phone: {phone}\n"

                # Add rating
                if rating := poi.get('rating', {}):
                    rating_value = rating.get('ratingValue', 'N/A')
                    rating_count = rating.get('ratingCount', 0)
                    formatted_results += f"Rating: {rating_value} ({rating_count} reviews)\n"

                # Add price range
                if price_range := poi.get('priceRange'):
                    formatted_results += f"Price Range: {price_range}\n"

                # Add hours
                if hours := poi.get('openingHours'):
                    formatted_results += f"Hours: {', '.join(hours)}\n"

                # Add description
                if desc_data.get('descriptions', {}).get(poi['id']):
                    formatted_results += f"Description: {desc_data['descriptions'][poi['id']]}\n"

                formatted_results += "\n"

            return formatted_results

    async def search(
        self,
        query: str,
        search_type: str = "web",
        num_results: Optional[int] = None,
        __event_emitter__=None
    ) -> str:
        """
        Perform a Brave search and return the results.

        Args:
            query: The search query string
            search_type: Type of search ('web' or 'local')
            num_results: Optional number of results to return (defaults to max_results valve)

        Returns:
            A formatted string containing the search results
        """
        try:
            # Input validation
            if not self.valves.brave_api_key:
                return "Error: Brave API key not configured. Please set up the API key in the tool settings."

            # Emit status that search is starting
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Initiating Brave search...", "done": False}
                })

            # Check rate limits
            await self._check_rate_limit()

            # Set number of results
            n_results = min(
                num_results if num_results is not None else self.valves.max_results,
                20  # Brave API maximum
            )

            # Perform the search
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Fetching search results...", "done": False}
                })

            if search_type.lower() == "local":
                result = await self._perform_local_search(query, n_results)
            else:
                result = await self._perform_web_search(query, n_results)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Search completed successfully", "done": True}
                })

            return result

        except Exception as e:
            error_message = f"An error occurred while performing the search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_message, "done": True}
                })
            return error_message
