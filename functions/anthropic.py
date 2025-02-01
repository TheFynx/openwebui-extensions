"""
title: Anthropic Pipe
version: 1.0.0
license: MIT
description: Improved Anthropic API integration with better error handling, rate limiting, and model management
author: TheFynx
author_url: https://github.com/thefynx
required_open_webui_version: 0.4.0
requirements: aiohttp>=3.8.0, structlog>=24.1.0
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- All Claude 3 models
- Streaming responses
- Image processing
- Function calling
- Rate limiting
- Error handling
- Resource usage tracking
- Response format control
- System prompt templates
- Response metadata
"""

import os
import json
import time
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, AsyncIterator, Any, Union
from dataclasses import dataclass
import aiohttp
import structlog

# Error Classes
class AnthropicError(Exception):
    """Base exception class for Anthropic-related errors."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(self.message)

class RateLimitError(AnthropicError):
    """Raised when rate limits are exceeded."""
    pass

class ValidationError(AnthropicError):
    """Raised for input validation failures."""
    pass

class AuthenticationError(AnthropicError):
    """Raised for authentication failures."""
    pass

class ModelNotFoundError(AnthropicError):
    """Raised when specified model is not found."""
    pass

class ServerError(AnthropicError):
    """Raised for server-side errors."""
    pass

# Resource Usage Tracking
@dataclass
class ResourceUsage:
    """Tracks resource usage metrics for a model."""
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    total_cost: float = 0.0
    processing_time: float = 0.0

class ResourceTracker:
    """Tracks and manages resource usage across models."""
    def __init__(self):
        self._usage: Dict[str, ResourceUsage] = {}
        self._lock = asyncio.Lock()
        self._price_per_1k_tokens = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.008, "output": 0.024},
            "claude-3-haiku": {"input": 0.003, "output": 0.012},
            "claude-3-5-sonnet": {"input": 0.008, "output": 0.024},
            "claude-3-5-haiku": {"input": 0.003, "output": 0.012},
        }

    async def track_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration: float
    ):
        """Record resource usage for a request."""
        async with self._lock:
            if model not in self._usage:
                self._usage[model] = ResourceUsage()

            usage = self._usage[model]
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.request_count += 1
            usage.processing_time += duration

            model_family = "-".join(model.split("-")[:-1])
            prices = self._price_per_1k_tokens.get(
                model_family,
                {"input": 0.008, "output": 0.024}
            )
            cost = (input_tokens / 1000 * prices["input"]) + (
                output_tokens / 1000 * prices["output"]
            )
            usage.total_cost += cost

    def get_usage(self, model: Optional[str] = None) -> Dict[str, ResourceUsage]:
        """Get resource usage statistics."""
        if model:
            return {model: self._usage.get(model, ResourceUsage())}
        return self._usage.copy()

# Rate Limiting
class TokenBucket:
    """Token bucket for rate limiting."""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int) -> bool:
        """Consume tokens from the bucket."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + time_passed * self.rate)
            self.last_update = now
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimitManager:
    """Manages rate limits for different models and operations."""
    def __init__(self, pipe: 'Pipe'):
        self.pipe = pipe
        self._limiters: Dict[str, Dict[str, TokenBucket]] = {}

    def get_limiter(self, model: str, limiter_type: str = "rpm") -> TokenBucket:
        """Get or create rate limiter for model and type."""
        if model not in self._limiters:
            self._limiters[model] = {}

        if limiter_type not in self._limiters[model]:
            limits = self.pipe.RATE_LIMITS[self.pipe.valves.RATE_LIMIT_TIER][model]
            rate = limits[limiter_type]
            self._limiters[model][limiter_type] = TokenBucket(
                rate=rate / 60,
                capacity=rate
            )

        return self._limiters[model][limiter_type]

    async def check_limits(self, model: str, estimated_tokens: int) -> None:
        """Check all applicable rate limits."""
        if not await self.get_limiter(model, "rpm").consume(1):
            raise RateLimitError("Request rate limit exceeded")

        if not await self.get_limiter(model, "input_tpm").consume(estimated_tokens):
            raise RateLimitError("Input token rate limit exceeded")

# Model Management
class ModelManager:
    """Manages model information and capabilities."""
    def __init__(self, pipe: 'Pipe'):
        self.pipe = pipe
        self._models_cache: Optional[Dict] = None
        self._cache_time: Optional[float] = None

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns available models in OpenWebUI format."""
        models_data = await self._fetch_models()
        available_models = []

        for model in models_data["data"]:
            model_id = model["id"]
            model_family = "-".join(model_id.split("-")[:-1])
            if model_family in self.pipe.RATE_LIMITS[self.pipe.valves.RATE_LIMIT_TIER]:
                available_models.append({
                    "id": f"anthropic/{model_id}",
                    "name": model_id,
                    "display_name": model.get("display_name", model_id),
                    "context_length": self._get_context_length(model_id),
                    "supports_vision": self._supports_vision(model_id),
                    "supports_tools": True
                })

        return available_models

    async def _fetch_models(self) -> Dict:
        """Fetch models from Anthropic API with caching."""
        if (
            self._models_cache is not None
            and self._cache_time is not None
            and (time.time() - self._cache_time) < self.pipe.valves.MODELS_CACHE_TTL
        ):
            return self._models_cache

        async with self.pipe._session.get(
            f"{self.pipe.valves.API_BASE_URL}/models",
            headers=self.pipe._get_headers()
        ) as response:
            if response.status != 200:
                raise ModelNotFoundError(
                    "Failed to fetch models",
                    status_code=response.status,
                    request_id=response.headers.get("x-request-id")
                )

            data = await response.json()
            self._models_cache = data
            self._cache_time = time.time()
            return data

    def _get_context_length(self, model_id: str) -> int:
        """Returns context length for given model."""
        return 200000  # All Claude 3 models support 200k context

    def _supports_vision(self, model_id: str) -> bool:
        """Checks if model supports vision."""
        return True  # All Claude 3 models support vision

class Pipe:
    """Enhanced Anthropic API integration for OpenWebUI."""

    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    REQUEST_TIMEOUT = (3.05, 60)

    # Rate limits per tier
    RATE_LIMITS = {
        1: {
            "claude-3-opus": {"rpm": 50, "input_tpm": 20000, "output_tpm": 4000},
            "claude-3-sonnet": {"rpm": 50, "input_tpm": 40000, "output_tpm": 8000},
            "claude-3-haiku": {"rpm": 50, "input_tpm": 50000, "output_tpm": 10000},
            "claude-3-5-sonnet": {"rpm": 50, "input_tpm": 40000, "output_tpm": 8000},
            "claude-3-5-haiku": {"rpm": 50, "input_tpm": 50000, "output_tpm": 10000},
        },
        2: {
            "claude-3-opus": {"rpm": 100, "input_tpm": 40000, "output_tpm": 8000},
            "claude-3-sonnet": {"rpm": 100, "input_tpm": 80000, "output_tpm": 16000},
            "claude-3-haiku": {"rpm": 100, "input_tpm": 100000, "output_tpm": 20000},
            "claude-3-5-sonnet": {"rpm": 100, "input_tpm": 80000, "output_tpm": 16000},
            "claude-3-5-haiku": {"rpm": 100, "input_tpm": 100000, "output_tpm": 20000},
        },
        3: {
            "claude-3-opus": {"rpm": 200, "input_tpm": 80000, "output_tpm": 16000},
            "claude-3-sonnet": {"rpm": 200, "input_tpm": 160000, "output_tpm": 32000},
            "claude-3-haiku": {"rpm": 200, "input_tpm": 200000, "output_tpm": 40000},
            "claude-3-5-sonnet": {"rpm": 200, "input_tpm": 160000, "output_tpm": 32000},
            "claude-3-5-haiku": {"rpm": 200, "input_tpm": 200000, "output_tpm": 40000},
        },
        4: {
            "claude-3-opus": {"rpm": 500, "input_tpm": 200000, "output_tpm": 40000},
            "claude-3-sonnet": {"rpm": 500, "input_tpm": 400000, "output_tpm": 80000},
            "claude-3-haiku": {"rpm": 500, "input_tpm": 500000, "output_tpm": 100000},
            "claude-3-5-sonnet": {"rpm": 500, "input_tpm": 400000, "output_tpm": 80000},
            "claude-3-5-haiku": {"rpm": 500, "input_tpm": 500000, "output_tpm": 100000},
        },
    }

    class Valves(BaseModel):
        """Configuration options (OpenWebUI's term for settings)."""
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Anthropic API key for authentication"
        )
        API_BASE_URL: str = Field(
            default="https://api.anthropic.com/v1",
            description="Anthropic API base URL"
        )
        RATE_LIMIT_TIER: int = Field(
            default=1,
            description="Rate limit tier (1-4)",
            ge=1,
            le=4
        )
        MAX_CONCURRENT: int = Field(
            default=50,
            description="Maximum number of concurrent requests"
        )
        MODELS_CACHE_TTL: int = Field(
            default=300,
            description="Cache time-to-live in seconds for models list"
        )
        RESPONSE_FORMAT: Optional[str] = Field(
            default=None,
            description="Default response format (json, markdown)"
        )
        SYSTEM_PROMPT_TEMPLATE: Optional[str] = Field(
            default=None,
            description="Default system prompt template"
        )
        METADATA_ENABLED: bool = Field(
            default=True,
            description="Include metadata in responses"
        )

        @validator("RESPONSE_FORMAT")
        def validate_response_format(cls, v):
            if v and v not in ["json", "markdown"]:
                raise ValueError("Response format must be 'json' or 'markdown'")
            return v

    def __init__(self):
        """Initialize the pipe with required components."""
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = structlog.get_logger()
        self._request_id: Optional[str] = None

        # Initialize components
        self.model_manager = ModelManager(self)
        self.rate_limit_manager = RateLimitManager(self)
        self.resource_tracker = ResourceTracker()

    async def setup(self):
        """Initialize resources."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(
                connect=self.REQUEST_TIMEOUT[0],
                total=self.REQUEST_TIMEOUT[1]
            )
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def cleanup(self):
        """Cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_headers(self, beta_features: Optional[List[str]] = None) -> Dict:
        """Generate request headers."""
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if beta_features:
            headers["anthropic-beta"] = ",".join(beta_features)
        return headers

    def _process_content(self, content: Any) -> List[Dict]:
        """Process message content."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed = []
        for item in content:
            if item["type"] == "text":
                processed.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed.append(self._process_image(item))
            elif item["type"] == "tool_calls":
                processed.append({
                    "type": "tool_calls",
                    "tool_calls": item["tool_calls"],
                    "cache_control": {"type": "ephemeral"},
                })
        return processed

    def _process_image(self, image_data: Dict) -> Dict:
        """Process image data."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValidationError(f"Unsupported media type: {media_type}")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": image_data["image_url"]["url"]},
        }

    async def _prepare_payload(self, body: Dict) -> Dict:
        """Prepare request payload."""
        model_name = body["model"].split("/")[-1]
        messages = []

        # Handle system message separately
        system_message = None
        filtered_messages = []

        for msg in body["messages"]:
            if msg["role"] == "system":
                system_message = msg["content"] if isinstance(msg["content"], str) else msg["content"][0]["text"]
            else:
                processed_content = self._process_content(msg["content"])
                filtered_messages.append({"role": msg["role"], "content": processed_content})

        payload = {
            "model": model_name,
            "messages": filtered_messages,
            "stream": True,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.7),
        }

        # Add system prompt from message, template, or direct input
        if system_message:
            payload["system"] = system_message
        elif self.valves.SYSTEM_PROMPT_TEMPLATE and "system" not in body:
            payload["system"] = self.valves.SYSTEM_PROMPT_TEMPLATE
        elif "system" in body:
            payload["system"] = body["system"]

        # Add response format if specified
        if self.valves.RESPONSE_FORMAT or "response_format" in body:
            payload["response_format"] = body.get("response_format", self.valves.RESPONSE_FORMAT)

        # Add tool definitions
        if "tools" in body:
            payload["tools"] = [
                {"type": "function", "function": tool}
                for tool in body["tools"]
            ]

        return payload

    async def pipe(
        self,
        body: Dict,
        __user__: Optional[Dict] = None,
        __event_emitter__: Optional[callable] = None,
    ) -> AsyncIterator[str]:
        """Main pipeline for processing requests."""
        if not self.valves.ANTHROPIC_API_KEY:
            raise AuthenticationError("Anthropic API key is required")

        try:
            if not self._session:
                await self.setup()

            model_name = body["model"].split("/")[-1]
            model_family = "-".join(model_name.split("-")[:-1])

            if model_family not in self.RATE_LIMITS[self.valves.RATE_LIMIT_TIER]:
                raise ModelNotFoundError(f"Unsupported model family: {model_family}")

            # Prepare request
            payload = await self._prepare_payload(body)
            estimated_input_tokens = len(str(payload)) // 4

            # Check rate limits
            await self.rate_limit_manager.check_limits(model_family, estimated_input_tokens)

            # Process request
            start_time = time.time()
            output_tokens = 0

            async with self._session.post(
                f"{self.valves.API_BASE_URL}/messages",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                self._request_id = response.headers.get("x-request-id")

                if response.status != 200:
                    error_text = await response.text()
                    raise ServerError(
                        f"API request failed: {error_text}",
                        status_code=response.status,
                        request_id=self._request_id,
                    )

                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data["type"] == "content_block_delta":
                                text = data["delta"]["text"]
                                new_tokens = len(text.split()) // 2
                                output_tokens += new_tokens

                                # Include metadata if enabled
                                if self.valves.METADATA_ENABLED and __event_emitter__:
                                    await __event_emitter__({
                                        "type": "metadata",
                                        "data": {
                                            "tokens": {
                                                "input": estimated_input_tokens,
                                                "output": output_tokens,
                                                "total": estimated_input_tokens + output_tokens
                                            },
                                            "model": model_name,
                                            "request_id": self._request_id
                                        }
                                    })

                                yield text

                            elif data["type"] == "message_stop":
                                # Track resource usage
                                duration = time.time() - start_time
                                await self.resource_tracker.track_request(
                                    model_family,
                                    estimated_input_tokens,
                                    output_tokens,
                                    duration,
                                )
                                break

                        except json.JSONDecodeError as e:
                            self.logger.warning("response_parse_error", error=str(e))
                            continue

        except Exception as e:
            error_msg = str(e)
            if isinstance(e, AnthropicError) and e.request_id:
                error_msg += f" (Request ID: {e.request_id})"

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Error: {error_msg}",
                        "done": True
                    }
                })
            yield f"Error: {error_msg}"

        finally:
            self._request_id = None

    # Define available models
    pipes = [
        {
            "id": f"anthropic/{name}",
            "name": name,
            "display_name": name,
            "context_length": 200000,
            "supports_vision": True,
        }
        for name in [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]
    ]
