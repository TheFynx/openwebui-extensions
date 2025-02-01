"""
title: GitHub Content Tool
description: Tool for fetching and searching content from GitHub repositories and gists
required_open_webui_version: 0.4.0
requirements: aiohttp, giturlparse
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
import aiohttp
import re
from urllib.parse import urlparse
from dataclasses import dataclass
import json

@dataclass
class GitHubRef:
    owner: str
    repo: str
    is_gist: bool = False

class Tools:
    class Valves(BaseModel):
        model_config = {
            "arbitrary_types_allowed": True
        }

        github_token: str = Field(
            "",
            description="GitHub Personal Access Token (optional, increases rate limits)"
        )
        max_file_size: int = Field(
            1000000,  # 1MB
            description="Maximum file size to fetch in bytes"
        )

    def __init__(self):
        """Initialize the Tool with valves."""
        self.valves = self.Valves()
        self.api_base = "https://api.github.com"

    def _parse_github_url(self, url: str) -> GitHubRef:
        """Parse different GitHub URL formats into owner and repo."""
        # Handle SSH format (git@github.com:user/repo)
        if url.startswith("git@"):
            ssh_pattern = r"git@github\.com:([^/]+)/([^.]+)(\.git)?"
            match = re.match(ssh_pattern, url)
            if match:
                return GitHubRef(owner=match.group(1), repo=match.group(2))

        # Handle HTTPS and browser URLs
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]

        # Handle gist URLs
        if "gist.github.com" in url:
            if len(path_parts) >= 2:
                return GitHubRef(owner=path_parts[0], repo=path_parts[1], is_gist=True)
            else:
                raise ValueError("Invalid Gist URL format")

        # Handle regular repository URLs
        if len(path_parts) >= 2:
            return GitHubRef(owner=path_parts[0], repo=path_parts[1].replace(".git", ""))

        raise ValueError("Invalid GitHub URL format")

    async def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenWebUI-GitHub-Tool"
        }
        if self.valves.github_token:
            headers["Authorization"] = f"token {self.valves.github_token}"
        return headers

    async def _fetch_gist_content(self, session: aiohttp.ClientSession, gist_id: str) -> str:
        """Fetch content from a GitHub Gist."""
        url = f"{self.api_base}/gists/{gist_id}"
        async with session.get(url, headers=await self._get_headers()) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"GitHub API error: {response.status} {error_text}")

            data = await response.json()
            result = "Gist Content:\n\n"
            for filename, file_info in data["files"].items():
                result += f"File: {filename}\n"
                result += "```\n"
                content = file_info.get("content", "Content not available")
                if len(content) > self.valves.max_file_size:
                    result += f"File too large (size: {len(content)} bytes)\n"
                else:
                    result += content
                result += "\n```\n\n"
            return result

    async def _fetch_repo_content(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        path: str = ""
    ) -> str:
        """Fetch content from a GitHub repository."""
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        async with session.get(url, headers=await self._get_headers()) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"GitHub API error: {response.status} {error_text}")

            data = await response.json()

            if isinstance(data, list):  # Directory
                result = f"Repository Contents ({path or 'root'}):\n\n"
                for item in data:
                    item_type = "📁" if item["type"] == "dir" else "📄"
                    result += f"{item_type} {item['path']}\n"
                return result
            else:  # Single file
                if data["size"] > self.valves.max_file_size:
                    return f"File too large (size: {data['size']} bytes)"

                if data["encoding"] == "base64":
                    import base64
                    content = base64.b64decode(data["content"]).decode('utf-8')
                    return f"File: {data['path']}\n```\n{content}\n```"
                return "Unsupported file encoding"

    async def _search_repo(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        query: str
    ) -> str:
        """Search within a specific repository."""
        url = f"{self.api_base}/search/code"
        params = {
            "q": f"{query} repo:{owner}/{repo}",
            "per_page": 10
        }

        async with session.get(
            url,
            headers=await self._get_headers(),
            params=params
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"GitHub API error: {response.status} {error_text}")

            data = await response.json()
            if data["total_count"] == 0:
                return "No matching files found."

            result = f"Search Results for '{query}':\n\n"
            for item in data["items"]:
                result += f"File: {item['path']}\n"
                result += f"URL: {item['html_url']}\n\n"

                # Fetch file content
                try:
                    content = await self._fetch_repo_content(session, owner, repo, item['path'])
                    result += f"{content}\n\n"
                except Exception as e:
                    result += f"Error fetching content: {str(e)}\n\n"

            return result

    async def fetch_content(
        self,
        url: str,
        search_query: Optional[str] = None,
        path: Optional[str] = "",
        __event_emitter__=None
    ) -> str:
        """
        Fetch content from a GitHub repository or gist.

        Args:
            url: GitHub URL (supports repository and gist URLs in various formats)
            search_query: Optional search query to filter content
            path: Optional path within repository to fetch specific content

        Returns:
            Formatted string containing the requested content
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Parsing GitHub URL...", "done": False}
                })

            ref = self._parse_github_url(url)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Fetching content...", "done": False}
                })

            async with aiohttp.ClientSession() as session:
                if ref.is_gist:
                    result = await self._fetch_gist_content(session, ref.repo)
                elif search_query:
                    result = await self._search_repo(session, ref.owner, ref.repo, search_query)
                else:
                    result = await self._fetch_repo_content(session, ref.owner, ref.repo, path)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Content retrieved successfully", "done": True}
                })

            return result

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_message, "done": True}
                })
            return error_message
