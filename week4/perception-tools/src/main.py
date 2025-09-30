"""
Main MCP server for perception tools.

This MCP server provides comprehensive perception capabilities including:
- Search tools (web search, knowledge base, file download)
- Multimodal understanding (web pages, documents, images, videos)
- File system operations (read, grep, summarization)
- Public data sources (weather, stocks, currency, Wikipedia, ArXiv, Wayback)
- Private data sources (Google Calendar, Notion)
"""
import logging
from dotenv import load_dotenv
from mcp.server import FastMCP
from pydantic import Field

# Import all tool functions
from search_tools import search_web, download_file, search_knowledge_base
from multimodal_tools import read_webpage, read_document, parse_image, parse_video
from filesystem_tools import read_file, grep_search, summarize_text
from public_data_tools import (
    get_weather, get_stock_price, convert_currency,
    search_wikipedia, search_arxiv, search_wayback
)
from private_data_tools import get_calendar_events, search_notion


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Initialize MCP server
mcp = FastMCP(
    "perception-tools",
    instructions="""
Perception Tools MCP Server

A comprehensive MCP server providing various perception and data retrieval capabilities:

## Search Tools
- Web search using Google Custom Search API
- Local knowledge base search
- File download from URLs

## Multimodal Understanding
- Web page content extraction
- Document reading (PDF, DOCX, PPTX)
- Image parsing and analysis
- Video metadata extraction

## File System Tools
- File reading with encoding support
- Grep-like pattern search
- Text summarization

## Public Data Sources
- Weather information
- Stock prices and market data
- Currency conversion
- Wikipedia search
- ArXiv academic papers
- Wayback Machine archives

## Private Data Sources
- Google Calendar events
- Notion workspace search
"""
)


# ============================================================================
# SEARCH TOOLS
# ============================================================================

@mcp.tool(description="Search the web using Google Custom Search API")
async def web_search(
    query: str = Field(description="Search query string"),
    num_results: int = Field(default=5, description="Number of results (1-10)"),
    language: str = Field(default="en", description="Language code"),
    country: str = Field(default="us", description="Country code")
):
    """Search the web and return results."""
    return await search_web(query, num_results, language, country)


@mcp.tool(description="Download a file from a URL to local storage")
async def download(
    url: str = Field(description="URL to download from"),
    output_path: str = Field(description="Local path to save the file"),
    overwrite: bool = Field(default=False, description="Overwrite existing file"),
    timeout: int = Field(default=180, description="Download timeout in seconds")
):
    """Download a file from URL."""
    return await download_file(url, output_path, overwrite, timeout)


@mcp.tool(description="Search a local knowledge base directory")
async def knowledge_base_search(
    query: str = Field(description="Search query"),
    knowledge_base_path: str = Field(description="Path to knowledge base directory"),
    top_k: int = Field(default=5, description="Number of top results")
):
    """Search local knowledge base."""
    return await search_knowledge_base(query, knowledge_base_path, top_k)


# ============================================================================
# MULTIMODAL UNDERSTANDING TOOLS
# ============================================================================

@mcp.tool(description="Read and extract content from a webpage")
async def webpage_reader(
    url: str = Field(description="URL of the webpage"),
    extract_text: bool = Field(default=True, description="Extract text content"),
    extract_links: bool = Field(default=False, description="Extract links")
):
    """Read webpage content."""
    return await read_webpage(url, extract_text, extract_links)


@mcp.tool(description="Read and extract content from documents (PDF, DOCX, PPTX)")
async def document_reader(
    file_path: str = Field(description="Path to document file or URL"),
    extract_images: bool = Field(default=False, description="Extract images")
):
    """Read document content."""
    return await read_document(file_path, extract_images)


@mcp.tool(description="Parse and analyze image files")
async def image_parser(
    image_path: str = Field(description="Path to image file or URL"),
    use_llm: bool = Field(default=True, description="Use LLM for analysis")
):
    """Parse image content."""
    return await parse_image(image_path, use_llm)


@mcp.tool(description="Parse and extract metadata from video files")
async def video_parser(
    video_path: str = Field(description="Path to video file or URL"),
    extract_frames: bool = Field(default=False, description="Extract sample frames"),
    frame_interval: int = Field(default=30, description="Frame extraction interval")
):
    """Parse video metadata."""
    return await parse_video(video_path, extract_frames, frame_interval)


# ============================================================================
# FILE SYSTEM TOOLS
# ============================================================================

@mcp.tool(description="Read a file and return its contents")
async def file_reader(
    file_path: str = Field(description="Path to the file"),
    encoding: str = Field(default="utf-8", description="File encoding"),
    max_length: int = Field(default=50000, description="Maximum characters to read")
):
    """Read file contents."""
    return await read_file(file_path, encoding, max_length)


@mcp.tool(description="Search for patterns in files (grep-like functionality)")
async def grep(
    pattern: str = Field(description="Regular expression pattern"),
    directory: str = Field(description="Directory to search in"),
    file_pattern: str = Field(default="*", description="File pattern (e.g., *.py)"),
    recursive: bool = Field(default=True, description="Search recursively"),
    case_sensitive: bool = Field(default=False, description="Case-sensitive search"),
    max_results: int = Field(default=100, description="Maximum results")
):
    """Search files for pattern."""
    return await grep_search(pattern, directory, file_pattern, recursive, case_sensitive, max_results)


@mcp.tool(description="Summarize long text content")
async def text_summarizer(
    text: str = Field(description="Text to summarize"),
    max_length: int = Field(default=500, description="Target summary length"),
    use_llm: bool = Field(default=True, description="Use LLM for summarization")
):
    """Summarize text."""
    return await summarize_text(text, max_length, use_llm)


# ============================================================================
# PUBLIC DATA SOURCE TOOLS
# ============================================================================

@mcp.tool(description="Get current weather information for a location")
async def weather(
    location: str = Field(description="City name or coordinates"),
    units: str = Field(default="metric", description="Units (metric/imperial/standard)")
):
    """Get weather data."""
    return await get_weather(location, units)


@mcp.tool(description="Get stock price and market information")
async def stock_price(
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)"),
    interval: str = Field(default="1d", description="Data interval")
):
    """Get stock price."""
    return await get_stock_price(symbol, interval)


@mcp.tool(description="Convert between currencies")
async def currency_converter(
    amount: float = Field(description="Amount to convert"),
    from_currency: str = Field(description="Source currency code (e.g., USD)"),
    to_currency: str = Field(description="Target currency code (e.g., EUR)")
):
    """Convert currency."""
    return await convert_currency(amount, from_currency, to_currency)


@mcp.tool(description="Search Wikipedia and get article summary")
async def wikipedia_search(
    query: str = Field(description="Search query"),
    language: str = Field(default="en", description="Wikipedia language"),
    sentences: int = Field(default=5, description="Summary sentence count")
):
    """Search Wikipedia."""
    return await search_wikipedia(query, language, sentences)


@mcp.tool(description="Search ArXiv for academic papers")
async def arxiv_search(
    query: str = Field(description="Search query"),
    max_results: int = Field(default=5, description="Maximum results"),
    sort_by: str = Field(default="relevance", description="Sort method")
):
    """Search ArXiv."""
    return await search_arxiv(query, max_results, sort_by)


@mcp.tool(description="Search Wayback Machine for archived web pages")
async def wayback_search(
    url: str = Field(description="URL to search for"),
    year: int | None = Field(default=None, description="Filter by year"),
    limit: int = Field(default=10, description="Maximum snapshots")
):
    """Search Wayback Machine."""
    return await search_wayback(url, year, limit)


# ============================================================================
# PRIVATE DATA SOURCE TOOLS
# ============================================================================

@mcp.tool(description="Get events from Google Calendar")
async def calendar_events(
    start_date: str | None = Field(default=None, description="Start date (ISO format)"),
    end_date: str | None = Field(default=None, description="End date (ISO format)"),
    calendar_id: str = Field(default="primary", description="Calendar ID"),
    max_results: int = Field(default=10, description="Maximum events")
):
    """Get calendar events."""
    return await get_calendar_events(start_date, end_date, calendar_id, max_results)


@mcp.tool(description="Search Notion workspace")
async def notion_search(
    query: str = Field(description="Search query"),
    database_id: str | None = Field(default=None, description="Specific database ID"),
    page_size: int = Field(default=10, description="Results per page")
):
    """Search Notion."""
    return await search_notion(query, database_id, page_size)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    logging.info("Starting Perception Tools MCP server!")
    mcp.run(transport="stdio")
