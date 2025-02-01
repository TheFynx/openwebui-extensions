"""
title: Gemini Pipe
version: 1.0.0
license: MIT
description: Google Gemini API integration with support for thinking models and image processing
author: TheFynx
author_url: https://github.com/thefynx
required_open_webui_version: 0.4.0
requirements: google-generativeai
environment_variables:
    - GOOGLE_API_KEY (required)

Supports:
- All Gemini models
- Streaming responses
- Image processing
- Thinking models with collapsible thoughts
- Status updates during processing
- Safety settings configuration
"""

import os
import re
import asyncio
import time
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator, Callable, Awaitable, Optional, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
import structlog

def to_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert messages to Google API content format."""
    contents = []
    for message in messages:
        if message.get("role") != "system":
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",", 1)[1] if "," in image_url else ""
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data,
                                }
                            })
                        else:
                            parts.append({"image_url": image_url})
                contents.append({"role": message.get("role"), "parts": parts})
            else:
                role = "user" if message.get("role") == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": content}],
                })
    return contents

# Error Classes
class GeminiError(Exception):
    """Base exception class for Gemini-related errors."""
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

class RateLimitError(GeminiError):
    """Raised when rate limits are exceeded."""
    pass

class ValidationError(GeminiError):
    """Raised for input validation failures."""
    pass

class AuthenticationError(GeminiError):
    """Raised for authentication failures."""
    pass

class ModelNotFoundError(GeminiError):
    """Raised when specified model is not found."""
    pass

class ServerError(GeminiError):
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
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
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

            prices = self._price_per_1k_tokens.get(
                model,
                {"input": 0.00025, "output": 0.0005}
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

        if not await self.get_limiter(model, "tpm").consume(estimated_tokens):
            raise RateLimitError("Token rate limit exceeded")


# Configuration Constants
DEBUG = False
MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_EMIT_INTERVAL = 5

class MessageContent(TypedDict):
    """Type definition for message content structure."""
    role: str
    parts: List[Dict[str, Any]]

class Pipe:
    """
    A pipe implementation for Google's Generative AI (Gemini) models.
    Handles model management, content generation, and streaming responses.
    """
    # Rate limits per tier
    RATE_LIMITS = {
        1: {  # Free tier
            "gemini-2.0-flash-exp": {"rpm": 10, "tpm": 4_000_000},
            "gemini-1.5-flash": {"rpm": 15, "tpm": 1_000_000},
            "gemini-1.5-flash-8b": {"rpm": 15, "tpm": 1_000_000},
            "gemini-1.5-pro": {"rpm": 2, "tpm": 32_000},
            "gemini-1.0-pro": {"rpm": 15, "tpm": 32_000},
            "text-embedding-004": {"rpm": 1500, "tpm": 1_000_000}, # Assuming a high tpm for embedding
            "aqa": {"rpm": 15, "tpm": 32_000}, # Assuming same as 1.0 pro for now
        },
        2: {  # Pay-as-you-go tier
            "gemini-2.0-flash-exp": {"rpm": 10, "tpm": 4_000_000},
            "gemini-1.5-flash": {"rpm": 2_000, "tpm": 4_000_000},
            "gemini-1.5-flash-8b": {"rpm": 4_000, "tpm": 4_000_000},
            "gemini-1.5-pro": {"rpm": 1_000, "tpm": 4_000_000},
            "gemini-1.0-pro": {"rpm": 360, "tpm": 120_000},
            "text-embedding-004": {"rpm": 1500, "tpm": 1_000_000}, # Assuming a high tpm for embedding
            "aqa": {"rpm": 360, "tpm": 120_000}, # Assuming same as 1.0 pro for now
        },
        3: { # Pay-as-you-go tier with higher limits
            "gemini-2.0-flash-exp": {"rpm": 10, "tpm": 4_000_000},
            "gemini-1.5-flash": {"rpm": 2_000, "tpm": 4_000_000},
            "gemini-1.5-flash-8b": {"rpm": 4_000, "tpm": 4_000_000},
            "gemini-1.5-pro": {"rpm": 1_000, "tpm": 4_000_000},
            "gemini-1.0-pro": {"rpm": 360, "tpm": 120_000},
            "text-embedding-004": {"rpm": 1500, "tpm": 1_000_000}, # Assuming a high tpm for embedding
            "aqa": {"rpm": 360, "tpm": 120_000}, # Assuming same as 1.0 pro for now
        }
    }
    def count_tokens(self, content: Union[str, List[Any]]) -> int:
        """
        Count tokens for a given content.

        Args:
            content: The content to count tokens for (text, list of messages, etc.)

        Returns:
            The total number of tokens in the content.
        """
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash") # Use a default model for token counting
            if isinstance(content, str):
                # Count tokens for text
                return model.count_tokens(content).total_tokens
            elif isinstance(content, list):
                # Count tokens for list of messages
                return model.count_tokens(to_contents(content)).total_tokens
            else:
                return 0
        except Exception as e:
            if DEBUG:
                print(f"[count_tokens] Error counting tokens: {e}")
            return 0
        finally:
            if DEBUG:
                print("[count_tokens] Completed token counting.")

    def _get_context_window(self, model: str) -> int:
        """Get context window for a given model."""
        if "gemini-1.5-flash" in model:
            return 1_000_000
        elif "gemini-1.5-pro" in model:
            return 2_000_000
        elif "gemini-1.0-pro" in model:
            return 32_000 # Assuming same as 1.5 pro for now
        else:
            return 8192 # Default context window

    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a given model."""
        if "gemini-1.5-flash" in model:
            return {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000}
        elif "gemini-1.5-pro" in model:
            return {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000}
        elif "gemini-1.0-pro" in model:
            return {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000} # Assuming same as 1.5 pro for now
        else:
            return {"input": 0.00025, "output": 0.0005} # Default pricing

    class Valves(BaseModel):
        """Configuration parameters for the Gemini pipe."""
        GOOGLE_API_KEY: str = Field(
            default="",
            description="API key for authenticating requests to the Google Generative AI API."
        )
        NAME_PREFIX: str = Field(
            default="GEMINI/",
            description="Prefix to be added before model names."
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False,
            description="Whether to use permissive safety settings for content generation."
        )
        THINKING_MODEL_PATTERN: str = Field(
            default=r"thinking",
            description="Regex pattern to identify thinking-enabled models."
        )
        EMIT_INTERVAL: int = Field(
            default=DEFAULT_EMIT_INTERVAL,
            description="Interval in seconds between status updates."
        )
        EMIT_STATUS_UPDATES: bool = Field(
            default=False,
            description="Whether to emit status updates during processing."
        )
        DEFAULT_SAFETY_SETTINGS: Dict[str, Any] = Field(
            default={},
            description="Default safety settings for content generation."
        )
        RATE_LIMIT_TIER: int = Field(
            default=1,
            description="Rate limit tier (1-3)",
            ge=1,
            le=3
        )
        MODELS_CACHE_TTL: int = Field(
            default=300,
            description="Cache time-to-live in seconds for models list"
        )
        METADATA_ENABLED: bool = Field(
            default=True,
            description="Include metadata in responses"
        )

        @validator("RATE_LIMIT_TIER")
        def validate_rate_limit_tier(cls, v):
            if v not in [1, 2, 3]:
                raise ValueError("Rate limit tier must be 1, 2, or 3")
            return v

    def __init__(self) -> None:
        """Initialize the Gemini pipe with default configuration."""
        try:
            self.id = "google_genai"
            self.type = "manifold"
            self.name = "Google: "
            self.logger = structlog.get_logger()
            self._request_id = None
            self._models_cache = None
            self._cache_time = None

            # Initialize components
            self.valves = self.Valves(
                **{
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                    "NAME_PREFIX": "GEMINI/",
                    "USE_PERMISSIVE_SAFETY": False,
                    "THINKING_MODEL_PATTERN": r"thinking",
                    "EMIT_INTERVAL": DEFAULT_EMIT_INTERVAL,
                    "EMIT_STATUS_UPDATES": os.getenv(
                        "EMIT_STATUS_UPDATES", "False"
                    ).lower() in ["true", "1", "yes"],
                    "DEFAULT_SAFETY_SETTINGS": {},
                    "RATE_LIMIT_TIER": int(os.getenv("GEMINI_RATE_LIMIT_TIER", "1")),
                    "MODELS_CACHE_TTL": 300,
                    "METADATA_ENABLED": True,
                }
            )

            self.rate_limit_manager = RateLimitManager(self)
            self.resource_tracker = ResourceTracker()

            if DEBUG:
                print("[INIT] Initialized Pipe with Valves configuration.")
                print(f"  EMIT_STATUS_UPDATES: {self.valves.EMIT_STATUS_UPDATES}")
                print(f"  RATE_LIMIT_TIER: {self.valves.RATE_LIMIT_TIER}")
        except Exception as e:
            if DEBUG:
                print(f"[INIT] Error during initialization: {e}")
            raise GeminiError(f"Initialization error: {str(e)}")
        finally:
            if DEBUG:
                print("[INIT] Initialization complete.")

    def _get_safety_settings(self, body: Dict[str, Any]) -> Dict[Any, Any]:
        """
        Get safety settings based on configuration.

        Args:
            body: Request body containing optional safety settings

        Returns:
            Dictionary of safety settings
        """
        if self.valves.USE_PERMISSIVE_SAFETY:
            if DEBUG:
                print("[_get_safety_settings] Using permissive safety settings.")
            return {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

        # Use safety settings from request body or fall back to defaults
        safety_settings = body.get("safety_settings", self.valves.DEFAULT_SAFETY_SETTINGS)
        if DEBUG:
            print("[_get_safety_settings] Using custom safety settings:", safety_settings)
        return safety_settings

    async def emit_thoughts(
        self,
        thoughts: str,
        __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Emit thoughts in a collapsible HTML element.

        Args:
            thoughts: The thought content to emit
            __event_emitter__: Callback function to emit events
        """
        try:
            if not thoughts.strip():
                if DEBUG:
                    print("[emit_thoughts] No thoughts to emit.")
                return
            enclosure = f"""<details>
<summary>Click to expand thoughts</summary>
{thoughts.strip()}
</details>""".strip()
            if DEBUG:
                print(f"[emit_thoughts] Emitting thoughts: {enclosure}")
            message_event = {
                "type": "message",
                "data": {"content": enclosure},
            }
            await __event_emitter__(message_event)
        except Exception as e:
            if DEBUG:
                print(f"[emit_thoughts] Error emitting thoughts: {e}")
        finally:
            if DEBUG:
                print("[emit_thoughts] Finished emitting thoughts.")

    def is_thinking_model(self, model_id: str) -> bool:
        """
        Check if the model is a thinking model based on the valve pattern.

        Args:
            model_id: The ID of the model to check

        Returns:
            True if the model is a thinking model, False otherwise
        """
        try:
            result = bool(
                re.search(self.valves.THINKING_MODEL_PATTERN, model_id, re.IGNORECASE)
            )
            if DEBUG:
                print(
                    f"[is_thinking_model] Model ID '{model_id}' is a thinking model: {result}"
                )
            return result
        except Exception as e:
            if DEBUG:
                print(f"[is_thinking_model] Error checking model: {e}")
            return False
        finally:
            if DEBUG:
                print("[is_thinking_model] Completed model check.")

    def get_google_models(self) -> List[Dict[str, str]]:
        """
        Retrieve available Google Gemini models.

        Returns:
            List of dictionaries containing model information with 'id' and 'name' keys.
            Returns error information if models cannot be fetched.
        """
        try:
            if not self.valves.GOOGLE_API_KEY:
                if DEBUG:
                    print("[get_google_models] GOOGLE_API_KEY is not set.")
                return [{"id": "error", "name": "GOOGLE_API_KEY is not set"}]

            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            models = genai.list_models()
            if DEBUG:
                print(f"[get_google_models] Retrieved {len(models)} models from Google.")

            available_models = [
                    {
                        "id": self.strip_prefix(model.name).replace("-latest", ""),
                        "name": f"{self.valves.NAME_PREFIX}{model.display_name}",
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    and model.name.startswith("models/")
                    and "gemini" in model.name.lower()
            ]

            # Add the text embedding model
            available_models.append({
                "id": "text-embedding-004",
                "name": f"{self.valves.NAME_PREFIX}text-embedding-004"
            })

            # Add the AQA model
            available_models.append({
                "id": "aqa",
                "name": f"{self.valves.NAME_PREFIX}aqa"
            })

            return available_models
        except Exception as e:
            if DEBUG:
                print(f"[get_google_models] Error fetching Google models: {e}")
            error_msg = f"Could not fetch models from Google: {str(e)}"
            if DEBUG:
                print(f"[get_google_models] {error_msg}")
            return [{"id": "error", "name": error_msg}]
        finally:
            if DEBUG:
                print("[get_google_models] Completed fetching Google models.")

    def strip_prefix(self, model_name: str) -> str:
        """
        Strip prefix from model name for standardization.

        Args:
            model_name: The full model name including prefix

        Returns:
            Cleaned model name without prefix
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            if DEBUG:
                print(f"[strip_prefix] Stripped prefix: '{stripped}' from '{model_name}'")
            return stripped
        except Exception as e:
            if DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails
        finally:
            if DEBUG:
                print("[strip_prefix] Completed prefix stripping.")

    def pipes(self) -> List[Dict[str, str]]:
        """
        Register and return all available Google models.

        Returns:
            List of dictionaries containing model information
        """
        try:
            models = self.get_google_models()
            if DEBUG:
                print(f"[pipes] Registered models: {models}")
            return models
        except Exception as e:
            if DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return []
        finally:
            if DEBUG:
                print("[pipes] Completed pipes method.")

    async def pipe(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> Union[str, Iterator[str]]:
        """
        Main pipe method to process incoming requests.

        Args:
            body: Request body containing model configuration and messages
            __event_emitter__: Optional callback for emitting events

        Returns:
            Generated content as string or iterator of strings if streaming
        """
        start_time = time.time()
        self._request_id = f"req_{int(start_time)}_{os.urandom(4).hex()}"

        try:
            if not self.valves.GOOGLE_API_KEY:
                raise AuthenticationError("GOOGLE_API_KEY is not set")

            try:
                genai.configure(api_key=self.valves.GOOGLE_API_KEY)
                if DEBUG:
                    print("[pipe] Configured Google Generative AI with API key.")
            except Exception as e:
                raise ServerError(f"Error configuring Google Generative AI: {str(e)}")

            model_id = body.get("model", "")
            if DEBUG:
                print(f"[pipe] Received model ID: '{model_id}'")

            try:
                model_id = self.strip_prefix(model_id)
                if DEBUG:
                    print(f"[pipe] Stripped model ID: '{model_id}'")
            except Exception as e:
                raise ValidationError(f"Error processing model ID: {str(e)}")

            if not model_id.startswith("gemini-") and model_id != "text-embedding-004" and model_id != "aqa":
                raise ModelNotFoundError(f"Invalid model name format: {model_id}")

            available_models = [model["id"] for model in self.get_google_models()]
            if model_id not in available_models:
                raise ModelNotFoundError(f"Model '{model_id}' not found in available models.")

            messages = body.get("messages", [])

            # Count input tokens
            try:
                estimated_input_tokens = self.count_tokens(messages)
            except Exception as e:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Error counting tokens: {str(e)}",
                            "done": True
                        }
                    })
                raise

            # Check context window
            context_window = self._get_context_window(model_id)
            if estimated_input_tokens > context_window:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Input tokens exceed context window limit ({context_window})",
                            "done": True
                        }
                    })
                raise ValidationError(f"Input tokens exceed context window limit ({context_window})")

            # Check rate limits
            try:
                await self.rate_limit_manager.check_limits(model_id, estimated_input_tokens)
            except RateLimitError as e:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Rate limit exceeded: {str(e)}",
                            "done": True
                        }
                    })
                raise

            stream = body.get("stream", False)

            if DEBUG:
                print(f"[pipe] Incoming messages: {messages}")
                print(f"[pipe] Stream mode: {stream}")

            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg.get("role") == "system"),
                None,
            )
            if DEBUG and system_message:
                print(f"[pipe] Extracted system message: '{system_message}'")

            contents = []
            try:
                for message in messages:
                    if message.get("role") != "system":
                        content = message.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for item in content:
                                if item.get("type") == "text":
                                    parts.append({"text": item.get("text", "")})
                                elif item.get("type") == "image_url":
                                    image_url = item.get("image_url", {}).get("url", "")
                                    if image_url.startswith("data:image"):
                                        image_data = (
                                            image_url.split(",", 1)[1]
                                            if "," in image_url
                                            else ""
                                        )
                                        parts.append(
                                            {
                                                "inline_data": {
                                                    "mime_type": "image/jpeg",
                                                    "data": image_data,
                                                }
                                            }
                                        )
                                    else:
                                        parts.append({"image_url": image_url})
                            contents.append(
                                {"role": message.get("role"), "parts": parts}
                            )
                        else:
                            role = "user" if message.get("role") == "user" else "model"
                            contents.append(
                                {
                                    "role": role,
                                    "parts": [{"text": content}],
                                }
                            )
                if DEBUG:
                    print(f"[pipe] Processed contents: {contents}")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error processing messages: {e}")
                return f"Error processing messages: {e}"

            # Insert system message at the beginning if present
            if system_message:
                try:
                    contents.insert(
                        0,
                        {
                            "role": "user",
                            "parts": [{"text": f"System: {system_message}"}],
                        },
                    )
                    if DEBUG:
                        print("[pipe] Inserted system message into contents.")
                except Exception as e:
                    if DEBUG:
                        print(f"[pipe] Error inserting system message: {e}")
                    return f"Error inserting system message: {e}"

            try:
                client = genai.GenerativeModel(model_name=model_id)
                if DEBUG:
                    print(f"[pipe] Initialized GenerativeModel with model ID: '{model_id}'")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error initializing GenerativeModel: {e}")
                return f"Error initializing GenerativeModel: {e}"

            generation_config = GenerationConfig(
                temperature=body.get("temperature", DEFAULT_TEMPERATURE),
                top_p=body.get("top_p", DEFAULT_TOP_P),
                top_k=body.get("top_k", DEFAULT_TOP_K),
                max_output_tokens=body.get("max_tokens", MAX_TOKENS),
                stop_sequences=body.get("stop", []),
            )

            try:
                safety_settings = self._get_safety_settings(body)
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error setting safety settings: {e}")
                return f"Error setting safety settings: {e}"

            if DEBUG:
                print("Google API request details:")
                print("  Model:", model_id)
                print("  Contents:", contents)
                print("  Generation Config:", generation_config)
                print("  Safety Settings:", safety_settings)
                print("  Stream:", stream)

            # Initialize timer variables
            thinking_timer_task: Optional[asyncio.Task] = None
            start_time: Optional[float] = None

            async def thinking_timer():
                """Asynchronous task to emit periodic status updates."""
                elapsed = 0
                try:
                    while True:
                        await asyncio.sleep(self.valves.EMIT_INTERVAL)
                        elapsed += self.valves.EMIT_INTERVAL
                        # Format elapsed time
                        if elapsed < 60:
                            time_str = f"{elapsed}s"
                        else:
                            minutes, seconds = divmod(elapsed, 60)
                            time_str = f"{minutes}m {seconds}s"
                        status_message = f"Thinking... ({time_str} elapsed)"
                        await emit_status(__event_emitter__, status_message, done=False)
                except asyncio.CancelledError:
                    if DEBUG:
                        print("[thinking_timer] Timer task cancelled.")
                except Exception as e:
                    if DEBUG:
                        print(f"[thinking_timer] Error in timer task: {e}")

            async def emit_status(event_emitter, message, done):
                """Emit status updates asynchronously."""
                try:
                    if self.valves.EMIT_STATUS_UPDATES and event_emitter:
                        status_event = {
                            "type": "status",
                            "data": {"description": message, "done": done},
                        }
                        if asyncio.iscoroutinefunction(event_emitter):
                            await event_emitter(status_event)
                        else:
                            # If the emitter is synchronous, run it in the event loop
                            loop = asyncio.get_event_loop()
                            loop.call_soon_threadsafe(event_emitter, status_event)
                        if DEBUG:
                            print(f"[emit_status] Emitted status: '{message}', done={done}")
                    else:
                        if DEBUG:
                            print(f"[emit_status] EMIT_STATUS_UPDATES is disabled. Skipping status: '{message}'")
                except Exception as e:
                    if DEBUG:
                        print(f"[emit_status] Error emitting status: {e}")
                finally:
                    if DEBUG:
                        print("[emit_status] Finished emitting status.")

            if self.is_thinking_model(model_id):
                try:
                    # Emit initial 'Thinking' status
                    if self.valves.EMIT_STATUS_UPDATES and __event_emitter__:
                        await emit_status(__event_emitter__, "Thinking...", done=False)

                    # Record the start time
                    start_time = time.time()

                    # Start the thinking timer
                    if self.valves.EMIT_STATUS_UPDATES:
                        thinking_timer_task = asyncio.create_task(thinking_timer())

                    # Define a helper function to call generate_content
                    def generate_content_sync(
                        client: genai.GenerativeModel,
                        contents: List[MessageContent],
                        generation_config: GenerationConfig,
                        safety_settings: Dict[Any, Any]
                    ) -> GenerateContentResponse:
                        return client.generate_content(
                            contents,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                        )

                    # Execute generate_content asynchronously to prevent blocking
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        generate_content_sync,
                        client,
                        contents,
                        generation_config,
                        safety_settings,
                    )

                    # Process response
                    if len(response.candidates[0].content.parts) > 1:
                        thoughts = response.candidates[0].content.parts[0].text
                        answer = response.candidates[0].content.parts[1].text

                        if __event_emitter__:
                            await self.emit_thoughts(thoughts, __event_emitter__)
                        result = answer
                    else:
                        result = response.candidates[0].content.parts[0].text

                    return result

                except Exception as e:
                    if DEBUG:
                        print(f"[pipe] Error during thinking model processing: {e}")
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {
                                "description": f"Error during thinking model processing: {e}",
                                "done": True
                            }
                        })
                    return f"Error: {e}"

                finally:
                    # Calculate total elapsed time
                    if start_time:
                        total_elapsed = int(time.time() - start_time)
                        if total_elapsed < 60:
                            total_time_str = f"{total_elapsed}s"
                        else:
                            minutes, seconds = divmod(total_elapsed, 60)
                            total_time_str = f"{minutes}m {seconds}s"

                        # Cancel the timer task
                        if thinking_timer_task:
                            thinking_timer_task.cancel()
                            try:
                                await thinking_timer_task
                            except asyncio.CancelledError:
                                if DEBUG:
                                    print("[pipe] Timer task successfully cancelled.")
                            except Exception as e:
                                if DEBUG:
                                    print(f"[pipe] Error cancelling timer task: {e}")

                        # Emit final status message
                        if self.valves.EMIT_STATUS_UPDATES:
                            final_status = f"Thinking completed in {total_time_str}."
                            await emit_status(__event_emitter__, final_status, done=True)

            # For non-thinking models or streaming
            else:
                output_tokens = 0
                if stream:
                    async def stream_generator():
                        """Asynchronous generator for streaming responses."""
                        nonlocal output_tokens, start_time
                        try:
                            response = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: client.generate_content(
                                    contents,
                                    generation_config=generation_config,
                                    safety_settings=safety_settings,
                                    stream=True,
                                )
                            )
                            for chunk in response:
                                if chunk.text:
                                    # Estimate tokens in chunk
                                    new_tokens = len(chunk.text.split()) // 2
                                    output_tokens += new_tokens

                                    # Emit metadata if enabled
                                    if self.valves.METADATA_ENABLED and __event_emitter__:
                                        await __event_emitter__({
                                            "type": "metadata",
                                            "data": {
                                                "tokens": {
                                                    "input": estimated_input_tokens,
                                                    "output": output_tokens,
                                                    "total": estimated_input_tokens + output_tokens
                                                },
                                                "model": model_id,
                                                "request_id": self._request_id
                                            }
                                        })

                                    yield chunk.text

                            # Track resource usage after completion
                            if start_time is not None:
                                duration = time.time() - start_time
                            else:
                                duration = 0
                            await self.resource_tracker.track_request(
                                model_id,
                                estimated_input_tokens,
                                output_tokens,
                                duration
                            )

                        except Exception as e:
                            error_msg = str(e)
                            if isinstance(e, GeminiError) and e.request_id:
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
                            if DEBUG:
                                print("[stream_generator] Stream generator completed.")

                    return stream_generator()
                else:
                    try:
                        response = client.generate_content(
                            contents,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                            stream=False,
                        )

                        # Estimate output tokens
                        output_tokens = len(response.text.split()) // 2

                        # Track resource usage
                        duration = time.time() - start_time
                        await self.resource_tracker.track_request(
                            model_id,
                            estimated_input_tokens,
                            output_tokens,
                            duration
                        )

                        if DEBUG:
                            print(f"[pipe] Received response: {response.text}")

                        # Emit final metadata
                        if self.valves.METADATA_ENABLED and __event_emitter__:
                            await __event_emitter__({
                                "type": "metadata",
                                "data": {
                                    "tokens": {
                                        "input": estimated_input_tokens,
                                        "output": output_tokens,
                                        "total": estimated_input_tokens + output_tokens
                                    },
                                    "model": model_id,
                                    "request_id": self._request_id,
                                    "duration": duration
                                }
                            })

                        async def single_response_generator():
                            """Asynchronous generator for single response."""
                            try:
                                yield response.text
                            except Exception as e:
                                error_msg = str(e)
                                if isinstance(e, GeminiError) and e.request_id:
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
                        return single_response_generator()
                    except Exception as e:
                        error_msg = str(e)
                        if isinstance(e, GeminiError) and e.request_id:
                            error_msg += f" (Request ID: {e.request_id})"
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {
                                    "description": f"Error: {error_msg}",
                                    "done": True
                                }
                            })
                        return f"Error: {error_msg}"

        except Exception as e:
            if DEBUG:
                print(f"[pipe] Error in pipe method: {e}")
            return f"Error: {e}"
        finally:
            if DEBUG:
                print("[pipe] Pipe method completed.")
