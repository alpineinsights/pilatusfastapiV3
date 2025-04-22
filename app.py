import pandas as pd
import os
import tempfile
import uuid
import time
import logging
from openai import OpenAI
from utils import QuartrAPI, AWSS3StorageHandler, TranscriptProcessor
import aiohttp
import asyncio
from typing import List, Dict, Tuple, Any, Optional
import json
import requests
from supabase_client import get_company_names, get_isin_by_name, get_quartrid_by_name, get_all_companies
import io
import re
import threading
import concurrent.futures
# Try to import PyMuPDF (fitz), but don't fail if it's not available
try:
    import fitz  # PyMuPDF
except ImportError:
    # Log warning instead of failing
    print("Warning: PyMuPDF (fitz) not installed. PDF generation functionality may be limited.")
from utils_helper import process_company_documents
from datetime import datetime
from logging_config import setup_logging
from logger import logger  # Import the configured logger
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenRouter client for all models
def initialize_openrouter():
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found")
        return None
    
    try:
        # Initialize the OpenAI client with OpenRouter base URL and API key
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        
        return client
    except Exception as e:
        logger.error(f"Error initializing OpenRouter client: {str(e)}")
        return None

# Extract valid JSON from Perplexity response
def extract_valid_json(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and returns only the valid JSON part from a Perplexity response object.
    
    Parameters:
        response (dict): The full API response object.

    Returns:
        dict: The parsed JSON object extracted from the content.
    
    Raises:
        ValueError: If no valid JSON can be parsed from the content.
    """
    # Navigate to the 'content' field
    content = (
        response
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    
    # Find the index of the closing </think> tag
    marker = "</think>"
    idx = content.rfind(marker)
    
    if idx == -1:
        # If marker not found, try parsing the entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("No </think> marker found and content is not valid JSON")
            # Return the raw content if it can't be parsed as JSON
            return {"content": content}
    
    # Extract the substring after the marker
    json_str = content[idx + len(marker):].strip()
    
    # Remove markdown code fence markers if present
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    if json_str.startswith("```"):
        json_str = json_str[3:].strip()
    if json_str.endswith("```"):
        json_str = json_str[:-3].strip()
    
    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse valid JSON from response content: {e}")
        # Return the raw content after </think> if it can't be parsed as JSON
        return {"content": json_str}

# Function to call Perplexity API
async def query_perplexity(query: str, company_name: str, conversation_context=None) -> Tuple[str, List[Dict]]:
    """Call Perplexity API with a financial analyst prompt for the specified company using OpenRouter"""
    
    client = initialize_openrouter()
    if not client:
        logger.error("OpenRouter client not initialized")
        return "Error: OpenRouter client not initialized", []
    
    try:
        logger.info(f"OpenRouter Perplexity API: Starting request for query about {company_name}")
        start_time = time.time()
        
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "Previous conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Create system prompt for financial analysis instructions only - USING ORIGINAL PROMPT
        system_prompt = "You are a senior financial analyst on listed equities. Give comprehensive and detailed responses. Refrain from mentioning or making comments on stock price movements. Do not make any buy or sell recommendation."
        
        # Create user message with research context and the original query - USING ORIGINAL FORMAT
        user_message = f"I am doing research on this listed company: {company_name}\n\n{query}\n\n{conversation_history}"
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call Perplexity API via OpenRouter
        logger.info("OpenRouter Perplexity API: Sending request")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="perplexity/sonar-reasoning-pro",
            messages=messages,
            temperature=0.2,  # Match original 
            max_tokens=2000   # Match original
        )
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenRouter Perplexity API: Received response in {api_time:.2f} seconds")
        
        content = response.choices[0].message.content
                
        # Extract citations if present
        citations = []
        
        # Check for citations in the content and extract them
        citation_regex = r'\[(.*?)\]\((https?://[^\s\)]+)\)'
        matches = re.findall(citation_regex, content)
        
        for i, (title, url) in enumerate(matches):
            citations.append({
                "id": i + 1,
                "title": title.strip(),
                "url": url.strip()
            })
        
        total_time = time.time() - start_time
        logger.info(f"OpenRouter Perplexity API: Total processing time: {total_time:.2f} seconds")
        
        return content, citations
    except Exception as e:
        logger.error(f"Error using OpenRouter Perplexity API: {str(e)}")
        
        error_details = str(e)
        if "rate limit" in error_details.lower() or "429" in error_details:
            return "Error: Rate limit exceeded. Please try again in a moment.", []
        
        return f"Error: {str(e)}", []

# Function to call Claude with combined outputs
def query_claude(query: str, company_name: str, gemini_output: str, perplexity_output: str, conversation_context=None) -> str:
    """Call Claude API with combined Gemini and Perplexity outputs for final synthesis using OpenRouter"""
    
    client = initialize_openrouter()
    if not client:
        logger.error("OpenRouter client not initialized")
        return "Error: OpenRouter client not initialized"
    
    try:
        logger.info("OpenRouter Claude API: Starting request")
        start_time = time.time()
        
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Create prompt for Claude - USING ORIGINAL PROMPT
        prompt = f"""You are a senior financial analyst on listed equities. Here is a question on {company_name}: {query}. 
Give a comprehensive and detailed response using ONLY the context provided below. Do not use your general knowledge or the Internet. 
If you encounter conflicting information between sources, prioritize the most recent source unless there's a specific reason not to (e.g., if the newer source explicitly references and validates the older information).
If the most recent available data is more than 6 months old, explicitly mention this in your response and caution that more recent developments may not be reflected in your analysis.
Refrain from mentioning or making comments on stock price movements. Do not make any buy or sell recommendation.{conversation_history}

Tone and format:
- Provide clear, detailed, and accurate information tailored to professional investors.
- When appropriate, for instance when the response involves a lot of figures, format your response in a table.
- If there are conflicting views or data points in different sources, acknowledge this and provide a balanced perspective.
- When appropriate, highlight any potential risks, opportunities, or trends that may not be explicitly stated in the query but are relevant to the analysis.
- If you don't have sufficient information to answer a query comprehensively, state this clearly and provide the best analysis possible with the available data.
- Recognize this might be a follow-up question to previous conversation. If so, provide a coherent response that acknowledges the conversation history.
- Be prepared to explain financial metrics, ratios, or industry-specific terms if requested.
- Maintain a professional and objective tone throughout your responses.

Remember, your goal is to provide valuable, data-driven insights that can aid professional investors in their decision-making process regarding the selected company, leveraging ONLY the provided context and NEVER using training data from your general knowledge.

Here is the context:

GEMINI OUTPUT (Based on company documents):
{gemini_output}

PERPLEXITY OUTPUT (Based on web search):
{perplexity_output}
"""
        
        # Create message structure
        messages = [
            {"role": "system", "content": "You are a senior financial analyst providing detailed analysis for professional investors."},
            {"role": "user", "content": prompt}
        ]
        
        # Call Claude API via OpenRouter
        logger.info("OpenRouter Claude API: Sending request")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            temperature=0.2,  # Match original
            max_tokens=4000    # Match original
        )
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenRouter Claude API: Received response in {api_time:.2f} seconds")
        
        # Extract the text content from the response
        final_response = response.choices[0].message.content
        
        total_time = time.time() - start_time
        logger.info(f"OpenRouter Claude API: Total processing time: {total_time:.2f} seconds")
        
        return final_response
    except Exception as e:
        logger.error(f"Error using OpenRouter Claude API: {str(e)}")
        
        error_details = str(e)
        if "rate limit" in error_details.lower() or "429" in error_details:
            return "Error: Rate limit exceeded. Please try again in a moment."
        
        return f"Error: {str(e)}"

# Function to download files from storage to temporary location
async def download_files_from_s3(file_urls: List[str]) -> List[str]:
    """Download files from AWS S3 storage to temporary location and return local paths"""
    try:
        aws_handler = AWSS3StorageHandler()
        temp_dir = tempfile.mkdtemp()
        local_files = []
        
        # Download files
        for file_url in file_urls:
            try:
                # Extract the S3 key from the URL
                parsed_url = urlparse(file_url)
                # Get the path portion of the URL
                path = parsed_url.path
                
                # Extract the key part (remove leading slash and bucket name if present)
                path_parts = path.split('/')
                if len(path_parts) > 2:  # Format: /bucket-name/key
                    s3_key = '/'.join(path_parts[2:])
                else:  # Format: /key or key
                    s3_key = path.lstrip('/')
                
                safe_filename = s3_key.replace('/', '-')
                local_path = os.path.join(temp_dir, safe_filename)
                
                logger.info(f"Downloading {s3_key} from AWS S3 storage to {local_path}")
                success = await aws_handler.download_file(s3_key, local_path)
                
                if success:
                    local_files.append(local_path)
                    logger.info(f"Successfully downloaded {s3_key} to {local_path}")
                else:
                    logger.error(f"Failed to download {s3_key}")
            except Exception as e:
                logger.error(f"Error downloading file: {str(e)}")
                
        return local_files
    except Exception as e:
        logger.error(f"Error in download_files_from_s3: {str(e)}")
        return []

# Function to query Gemini with file context
async def query_gemini_async(query: str, file_paths: List[str], conversation_context=None) -> str:
    """Query Gemini model with context from files (async version)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: query_gemini(query, file_paths, conversation_context))

def query_gemini(query: str, file_paths: List[str], conversation_context=None) -> str:
    """Query Gemini model with context from files using OpenRouter"""
    try:
        logger.info(f"OpenRouter Gemini API: Starting analysis with {len(file_paths)} documents")
        start_time = time.time()
        
        # Make sure OpenRouter client is initialized
        client = initialize_openrouter()
        if not client:
            return "Error initializing OpenRouter client"
        
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "Previous conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Prepare contents for messages
        contents = []
        file_processing_time = time.time()
        
        # Process files
        logger.info("OpenRouter Gemini API: Processing document files")
        for file_path in file_paths:
            try:
                # Open the file for binary reading
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    
                # Add file as a base64-encoded part for message content
                import base64
                base64_data = base64.b64encode(file_data).decode("utf-8")
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{base64_data}"
                    }
                })
            except Exception as e:
                logger.error(f"Error processing file for OpenRouter Gemini: {str(e)}")
        
        # Add text content part - USING ORIGINAL PROMPT FORMAT
        contents.append({
            "type": "text",
            "text": f"You are a senior financial analyst. Review the attached documents and provide a detailed and structured answer to the user's query. User's query: '{query}'\n\n{conversation_history}"
        })
        
        # Create message structure
        messages = [{"role": "user", "content": contents}]
        
        file_processing_time = time.time() - file_processing_time
        logger.info(f"OpenRouter Gemini API: Processed {len(file_paths)} files in {file_processing_time:.2f} seconds")
        
        if not contents:
            return "No files were successfully processed for OpenRouter Gemini"
        
        # Generate content with files as context
        logger.info("OpenRouter Gemini API: Sending request to API")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=messages,
            max_tokens=7000,
            temperature=0.2  # Match original temperature
        )
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenRouter Gemini API: Received response in {api_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"OpenRouter Gemini API: Total processing time: {total_time:.2f} seconds")
        
        # Return the response text
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying OpenRouter Gemini: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"
