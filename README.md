# OpenWebUI Extensions

A personal collection of extensions for [OpenWebUI](https://github.com/open-webui/open-webui) that I use across my systems. These extensions are shared publicly for anyone interested in extending their OpenWebUI installation.

## What's Included

- **Functions**: Advanced model integrations and content processing:
  - Anthropic Integration: Enhanced Claude 3 API support
  - Google Gemini Integration: Complete Gemini model support
  - Artifacts: Interactive HTML content visualization
- **Tools**: Additional capabilities including:
  - Brave Search: Web and local search integration
  - GitHub Content: Repository and gist content fetching/searching
  - Time Tools: Date and time utilities
- **Prompts**: Collection of custom prompts

## Why This Exists

I created this repository to:

1. Maintain consistency across my different OpenWebUI installations
2. Share useful extensions with the OpenWebUI community
3. Make it easier to track and version my custom extensions

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/thefynx/openwebui-extensions.git
    ```

2. Install the package:

    ```bash
    pip install -e .
    ```

## Requirements

- OpenWebUI ≥ 0.3.17
- Python ≥ 3.8

## Directory Structure

```bash
.
├── functions/          # Model integrations and core functionality
│   ├── anthropic.py   # Anthropic API integration
│   └── gemini.py      # Google Gemini integration
├── tools/             # Additional capabilities
│   ├── brave-search.py  # Brave Search integration
│   ├── github-content.py # GitHub content fetching
│   └── get-time.py      # Time utilities
└── prompts/           # Custom prompt collections
```

## Functions

### Anthropic Integration

Enhanced Claude 3 API integration with comprehensive features:

- **Model Support**:
  - All Claude 3 models (Opus, Sonnet, Haiku)
  - Context length up to 200k tokens
  - Vision capabilities for all models
  - Function calling support

- **Advanced Features**:
  - Streaming responses with real-time token counting
  - Automatic rate limiting based on tier (1-4)
  - Resource usage tracking (tokens, costs, processing time)
  - Response format control (JSON, Markdown)
  - System prompt templates
  - Response metadata
  - Error handling with detailed feedback

- **Configuration**:
  - Rate limit tiers with different RPM and TPM limits
  - Configurable API endpoints
  - Safety settings management
  - Model caching for performance
  - Concurrent request handling

### Google Gemini Integration

Complete Google Gemini API integration with extensive capabilities:

- **Model Support**:
  - All Gemini models (1.0, 1.5, 2.0)
  - Flash and Pro variants
  - Vision capabilities
  - Text embeddings
  - AQA (Automated Question Answering)

- **Key Features**:
  - Streaming responses
  - Thinking models with collapsible thoughts
  - Status updates during processing
  - Rate limiting with tier support (1-3)
  - Resource tracking and cost management
  - Safety settings configuration
  - Error handling and recovery

- **Configuration**:
  - Customizable rate limit tiers
  - Safety settings management
  - Status update intervals
  - Model caching
  - Response metadata

## Tools

### Brave Search

Enables web and local search capabilities using the Brave Search API:

- Web search with customizable result count
- Local search for businesses and places
- Rate limiting and error handling
- Configurable via admin settings

### GitHub Content

Fetches and searches content from GitHub repositories and gists:

- Supports multiple URL formats (HTTPS, SSH, Gist)
- Repository content browsing
- Gist content fetching
- Code search within repositories
- Configurable GitHub token for increased rate limits

### Time Tools

Provides date and time utilities:

- Current date and time retrieval
- Configurable time format (12/24 hour)
- User-configurable display preferences
