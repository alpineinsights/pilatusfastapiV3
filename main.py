from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import asyncio
import aiohttp
import time
import logging
import json
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from utils import QuartrAPI, AWSS3StorageHandler, TranscriptProcessor
from supabase_client import get_quartrid_by_name
from logger import logger
from urllib.parse import urlparse, unquote  # For parsing citation URLs
import tempfile
import base64
import unicodedata

# Load environment variables
load_dotenv()

# Configure logging
logger.info("Starting FastAPI Financial Insights Application")

# Initialize FastAPI app
app = FastAPI(
    title="Financial Insights API",
    description="API for generating financial insights about companies using a multi-LLM pipeline",
    version="1.0.0"
)

# Input model
class QueryRequest(BaseModel):
    company_name: str
    query: str
    conversation_context: Optional[List[Dict[str, str]]] = None

# Response model
class QueryResponse(BaseModel):
    answer: str
    processing_time: float
    sources: Optional[Dict[str, List[Dict[str, str]]]] = None

# Global conversation context - will be updated per company
conversation_contexts = {}

# Initialize OpenRouter client for all models
def initialize_openrouter():
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    
    # Log the loaded API key (or lack thereof)
    if OPENROUTER_API_KEY:
        logger.info(f"OpenRouter API Key loaded successfully (length: {len(OPENROUTER_API_KEY)})")
    else:
        logger.error("OpenRouter API key NOT FOUND in environment variables")
        return None
    
    try:
        # Initialize the OpenAI client with OpenRouter base URL and API key
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        logger.info("OpenAI client initialized for OpenRouter")
        return client
    except Exception as e:
        logger.error(f"Error initializing OpenRouter client: {str(e)}")
        return None

# Remove redundant initializer functions
# Initialize Gemini model - keeping for backward compatibility
def initialize_gemini():
    return initialize_openrouter()

# Initialize Claude client - keeping for backward compatibility 
def initialize_claude():
    return initialize_openrouter()

# Extract valid JSON from Perplexity response
def extract_valid_json(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and returns only the valid JSON part from a Perplexity response object.
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
                
        # Try to extract citations from the response (both ways)
        citations = []
        
        # Method 1: Try to get citations directly from response object if available
        # Convert response to dict to check if citations field exists
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}
        direct_citations = response_dict.get('citations', [])
        
        if direct_citations:
            logger.info(f"Found {len(direct_citations)} citations directly in the response")
            citations = direct_citations
        else:
            # Method 2: Extract citations from the text using regex (markdown links)
            logger.info("No direct citations found, extracting from text using regex")
            citation_regex = r'\[(.*?)\]\((https?://[^\s\)]+)\)'
            matches = re.findall(citation_regex, content)
            
            for i, (title, url) in enumerate(matches):
                citations.append({
                    "id": i + 1,
                    "title": title.strip(),
                    "url": url.strip()
                })
            
            logger.info(f"Extracted {len(citations)} citations from text")
        
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
        logger.info("Starting Claude synthesis process")
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
        
        # Try up to 3 times to get a valid response from Claude
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Call Claude API via OpenRouter (only log first attempt or errors)
                if attempt == 1:
                    logger.info("Calling Claude API via OpenRouter")
                
                response = client.chat.completions.create(
                    model="anthropic/claude-3.7-sonnet",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4000
                )
                
                # Check if response has the expected structure
                if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and response.choices[0].message:
                        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                            final_response = response.choices[0].message.content
                            logger.info(f"Claude synthesis completed successfully ({len(final_response)} chars)")
                            # Success! Break out of retry loop
                            break
                        else:
                            logger.error(f"Claude response missing content (attempt {attempt}/{max_retries})")
                    else:
                        logger.error(f"Claude response missing message (attempt {attempt}/{max_retries})")
                else:
                    # Only log raw response on the last attempt to avoid log bloat
                    if attempt == max_retries:
                        logger.error(f"Claude response structure invalid. Raw response: {response}")
                    else:
                        logger.error(f"Claude response structure invalid (attempt {attempt}/{max_retries})")
                
                # If we get here, the response was not valid. If this was the last attempt, set a default error message
                if attempt == max_retries:
                    final_response = "Our synthesis system encountered technical difficulties. Please try again in a few moments."
                # Otherwise, wait a bit before retrying
                else:
                    logger.info(f"Retrying Claude API (attempt {attempt}/{max_retries})")
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Claude API error (attempt {attempt}/{max_retries}): {str(e)}")
                # If this was the last attempt, set a default error message
                if attempt == max_retries:
                    final_response = f"Error connecting to our analysis service. Please try again later."
                # Otherwise, wait a bit before retrying
                else:
                    time.sleep(2)
        
        logger.info(f"Claude process completed in {time.time() - start_time:.1f}s")
        return final_response
    except Exception as e:
        logger.error(f"Unexpected error in Claude function: {str(e)}")
        return f"An unexpected error occurred. Please try again later."

# Function to process company documents and generate embeddings
async def process_company_documents(company_id: str, company_name: str, event_type: str = "all") -> List[Dict]:
    """Process company documents and return list of file information"""
    try:
        async with aiohttp.ClientSession() as session:
            # Initialize API and handlers
            quartr_api = QuartrAPI()
            storage_handler = AWSS3StorageHandler()
            transcript_processor = TranscriptProcessor()
            
            # Get company data from Quartr API using company ID
            company_data = await quartr_api.get_company_events(company_id, session, event_type)
            if not company_data:
                logger.error(f"Failed to get company data for ID: {company_id}")
                return []
            
            logger.info(f"Processing documents for company: {company_name} (ID: {company_id})")
                
            events = company_data.get('events', [])
            if not events:
                logger.warning(f"No events found for company: {company_name} (ID: {company_id})")
                return []
                
            # Sort events by date (descending)
            events.sort(key=lambda x: x.get('eventDate', ''), reverse=True)
            
            processed_files = []
            transcript_count = 0
            report_count = 0
            pdf_count = 0
            
            # Process up to 2 documents of each type
            for event in events:
                # Stop processing if we have enough documents (2 of each type)
                if transcript_count >= 2 and report_count >= 2 and pdf_count >= 2:
                    break
                    
                event_date = event.get('eventDate', '').split('T')[0] if 'T' in event.get('eventDate', '') else event.get('eventDate', '')
                event_title = event.get('eventTitle', event.get('title', 'Unknown Event'))
                
                # Log event details for debugging
                logger.info(f"Processing event: {event_title} from {event_date}")
                
                # Process PDF/slides (if we need more)
                if pdf_count < 2 and event.get('pdfUrl'):
                    try:
                        # Log the URL we're trying to download
                        logger.info(f"Attempting to download slides from: {event.get('pdfUrl')}")
                        
                        async with session.get(event.get('pdfUrl')) as response:
                            if response.status == 200:
                                content = await response.read()
                                original_filename = event.get('pdfUrl').split('/')[-1]
                                
                                # Remove any URL query parameters from the original filename
                                if '?' in original_filename:
                                    original_filename = original_filename.split('?')[0]
                                
                                filename = storage_handler.create_filename(
                                    company_name, event_date, event_title, 'slides', original_filename
                                )
                                
                                success = await storage_handler.upload_file(
                                    content, filename, 'application/pdf'
                                )
                                
                                if success:
                                    public_url = storage_handler.get_public_url(filename)
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'presentation',
                                        'title': event_title,
                                        'date': event_date,
                                        'url': public_url
                                    })
                                    pdf_count += 1
                                    logger.info(f"Successfully processed and stored slides: {filename}")
                            else:
                                logger.error(f"Failed to download slides: HTTP {response.status}")
                    except Exception as e:
                        logger.error(f"Error processing slides for {event_title}: {str(e)}")
                
                # Process report (if we need more)
                if report_count < 2 and event.get('reportUrl'):
                    try:
                        # Log the URL we're trying to download
                        logger.info(f"Attempting to download report from: {event.get('reportUrl')}")
                        
                        async with session.get(event.get('reportUrl')) as response:
                            if response.status == 200:
                                content = await response.read()
                                original_filename = event.get('reportUrl').split('/')[-1]
                                
                                # Remove any URL query parameters from the original filename
                                if '?' in original_filename:
                                    original_filename = original_filename.split('?')[0]
                                
                                filename = storage_handler.create_filename(
                                    company_name, event_date, event_title, 'report', original_filename
                                )
                                
                                success = await storage_handler.upload_file(
                                    content, filename, 'application/pdf'
                                )
                                
                                if success:
                                    public_url = storage_handler.get_public_url(filename)
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'report',
                                        'title': event_title,
                                        'date': event_date,
                                        'url': public_url
                                    })
                                    report_count += 1
                                    logger.info(f"Successfully processed and stored report: {filename}")
                            else:
                                logger.error(f"Failed to download report: HTTP {response.status}")
                    except Exception as e:
                        logger.error(f"Error processing report for {event_title}: {str(e)}")
                
                # Process transcript (if we need more)
                if transcript_count < 2 and event.get('transcriptUrl'):
                    try:
                        # Log the transcript URL we're processing
                        logger.info(f"Processing transcript from: {event.get('transcriptUrl')}")
                        
                        # Get transcript data
                        transcripts = event.get('transcripts', {})
                        if not transcripts:
                            # If the transcripts object is empty, check for liveTranscripts
                            transcripts = event.get('liveTranscripts', {})
                        
                        transcript_text = await transcript_processor.process_transcript(
                            event.get('transcriptUrl'), transcripts, session
                        )
                        
                        if transcript_text:
                            pdf_data = transcript_processor.create_pdf(
                                company_name, event_title, event_date, transcript_text
                            )
                            
                            filename = storage_handler.create_filename(
                                company_name, event_date, event_title, 'transcript', 'transcript.pdf'
                            )
                            
                            success = await storage_handler.upload_file(
                                pdf_data, filename, 'application/pdf'
                            )
                            
                            if success:
                                public_url = storage_handler.get_public_url(filename)
                                processed_files.append({
                                    'filename': filename,
                                    'type': 'transcript',
                                    'title': event_title,
                                    'date': event_date,
                                    'url': public_url,
                                    'text': transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text
                                })
                                transcript_count += 1
                                logger.info(f"Successfully processed and stored transcript: {filename}")
                    except Exception as e:
                        logger.error(f"Error processing transcript for {event_title}: {str(e)}")
            
            # Log the number of documents processed
            logger.info(f"Processed {pdf_count} presentations, {report_count} reports, and {transcript_count} transcripts")
            return processed_files
    except Exception as e:
        logger.error(f"Error in process_company_documents: {str(e)}")
        return []

# Function to download files from S3 to local storage
async def download_files_from_s3(file_urls: List[str]) -> List[str]:
    """Download files from AWS S3 storage to temporary location and return local paths"""
    try:
        logger.info(f"Downloading {len(file_urls)} files from source URLs")
        temp_dir = tempfile.mkdtemp()
        local_files = []
        
        # Download files using a synchronous approach with direct HTTP requests
        for i, file_url in enumerate(file_urls):
            try:
                # Extract the filename from the URL for local storage
                from urllib.parse import urlparse, unquote
                parsed_url = urlparse(file_url)
                path = unquote(parsed_url.path)
                
                # Transliterate accented characters in the filename
                def transliterate(text):
                    # Normalize to decomposed form (separate accents from letters)
                    text = unicodedata.normalize('NFKD', text)
                    # Remove diacritical marks (accents)
                    text = ''.join([c for c in text if not unicodedata.combining(c)])
                    # Replace any remaining non-ASCII characters
                    text = re.sub(r'[^\x00-\x7F]+', '_', text)
                    return text
                
                # Create safe local filename
                basename = os.path.basename(path)
                filename = transliterate(basename)
                if not filename:
                    filename = f"file_{i}.pdf"
                
                # Create a safe local path
                local_path = os.path.join(temp_dir, filename)
                
                # Create directories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Use a direct HTTP GET request instead of S3 client
                async with aiohttp.ClientSession() as session:
                    logger.info(f"Downloading from URL: {file_url}")
                    async with session.get(file_url) as response:
                        if response.status == 200:
                            # Read the content
                            content = await response.read()
                            content_length = len(content)
                            
                            if content_length == 0:
                                logger.error(f"File content is empty: {file_url}")
                                continue
                                
                            # Write to local file
                            with open(local_path, 'wb') as f:
                                f.write(content)
                            
                            # Verify file was written properly
                            if os.path.exists(local_path):
                                file_size = os.path.getsize(local_path)
                                if file_size > 0:
                                    local_files.append(local_path)
                                    logger.info(f"Downloaded {filename} ({file_size} bytes)")
                                else:
                                    logger.error(f"File exists but is empty: {local_path}")
                            else:
                                logger.error(f"File failed to download: {local_path}")
                        else:
                            logger.error(f"HTTP error {response.status} for URL: {file_url}")
            except Exception as e:
                logger.error(f"Error downloading {file_url}: {str(e)}")
                
        logger.info(f"Downloaded {len(local_files)}/{len(file_urls)} files successfully")
        return local_files
    except Exception as e:
        logger.error(f"Error in download process: {str(e)}")
        return []

# Function to analyze documents with Gemini
async def analyze_documents_with_gemini(company_name: str, query: str, processed_files: List[Dict], conversation_context=None):
    """Analyze company documents using Gemini AI via OpenRouter"""
    logger.info(f"Analyzing documents for {company_name} with Gemini (files: {len(processed_files)})")
    
    client = initialize_openrouter()
    if not client:
        logger.error("OpenRouter client not initialized")
        return "Error: OpenRouter client not initialized"
    
    try:
        # Build conversation history for context
        conversation_history = ""
        if conversation_context:
            conversation_history = "Previous conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Download files from storage
        file_urls = [doc['url'] for doc in processed_files if 'url' in doc]
        local_files = await download_files_from_s3(file_urls)
        
        # Log file details for debugging
        total_file_size = 0
        for file_path in local_files:
            file_size = os.path.getsize(file_path)
            total_file_size += file_size
            logger.info(f"File: {os.path.basename(file_path)}, Size: {file_size} bytes")
        logger.info(f"Total size of all files: {total_file_size} bytes")
        
        if not local_files:
            logger.warning("No files were successfully downloaded for analysis")
            # Fallback to text-only if files couldn't be downloaded
            documents_text = "No document contents available."
            
            # Create a text-only prompt as fallback
            prompt = f"""You are a financial analyst assistant specialized in analyzing company financial documents. 

Task: Please analyze the provided company documents for {company_name} and answer the following query: {query}

Base your analysis EXCLUSIVELY on the documents provided below. If the information isn't in the documents, state that clearly.
{conversation_history}

Here are the documents:
{documents_text}
"""
            
            # Create message structure for text-only fallback
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            # Prepare contents for messages with file data
            contents = []
            
            # Process files
            for file_path in local_files:
                try:
                    # Open the file for binary reading
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        
                    # Check file data integrity
                    if len(file_data) == 0:
                        logger.error(f"File is empty: {file_path}")
                        continue
                        
                    # Add file as a base64-encoded part for message content
                    base64_data = base64.b64encode(file_data).decode("utf-8")
                    file_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{base64_data}"
                        }
                    }
                    contents.append(file_content)
                    logger.info(f"Added file to message: {os.path.basename(file_path)}, encoded size: {len(base64_data)} chars")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
            
            # Add text content as the last part of the message
            text_prompt = f"""You are a financial analyst assistant specialized in analyzing company financial documents. 
Task: Please analyze the attached documents for {company_name} and answer the following query: {query}

Base your analysis EXCLUSIVELY on the attached documents. If the information isn't in the documents, state that clearly.

{conversation_history}"""
            
            text_content = {
                "type": "text",
                "text": text_prompt
            }
            contents.append(text_content)
            logger.info(f"Added text prompt to message with {len(contents)-1} attached documents")
            
            # Ensure we have properly formatted content for OpenRouter/Gemini
            if len(contents) <= 1:
                logger.error("No document content was successfully added to the message")
                return "Error: Failed to process document content for analysis."
                
            # Create message structure for OpenRouter/Gemini
            messages = [{"role": "user", "content": contents}]
            
            # Log the structure for debugging (without the base64 data which is too long)
            debug_contents = []
            for item in contents:
                if item.get("type") == "image_url":
                    debug_contents.append({
                        "type": "image_url",
                        "image_url": {"url": "data:application/pdf;base64,[BASE64_DATA]"}
                    })
                else:
                    debug_contents.append(item)
            logger.info(f"Message structure: {debug_contents}")
        
        # Call Gemini API via OpenRouter
        logger.info("Calling Gemini API")
        api_start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=messages,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Extract the text content from the response
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and response.choices[0].message:
                    if hasattr(response.choices[0].message, 'content'):
                        final_response = response.choices[0].message.content
                        api_time = time.time() - api_start_time
                        logger.info(f"Gemini analysis completed in {api_time:.1f}s, response length: {len(final_response)} chars")
                        
                        # If response is very short and completed quickly, it might be an error
                        if len(final_response) < 500 and api_time < 5.0:
                            logger.warning(f"Suspiciously short response ({len(final_response)} chars) and quick processing time ({api_time:.1f}s). Response: {final_response}")
                        
                        return final_response
                    else:
                        logger.error("Gemini response missing content")
                else:
                    logger.error("Gemini response missing message")
            else:
                logger.error("Gemini response missing choices")
                
            # If we get here, the response parsing failed
            api_time = time.time() - api_start_time
            logger.error(f"Failed to extract content from Gemini response after {api_time:.1f}s")
            return "Error: Failed to extract content from Gemini response."
            
        except Exception as api_error:
            api_time = time.time() - api_start_time
            logger.error(f"Gemini API error after {api_time:.1f}s: {str(api_error)}")
            return f"Error calling Gemini API: {str(api_error)}"
            
    except Exception as e:
        logger.error(f"Error in document analysis: {str(e)}")
        return f"Error: {str(e)}"

# Main endpoint for financial insights
@app.post("/api/insights", response_model=QueryResponse)
async def get_financial_insights(request: QueryRequest):
    start_time = time.time()
    logger.info(f"Received request for company: {request.company_name}, query: {request.query}")
    
    try:
        # Get company ID from Supabase
        company_id = get_quartrid_by_name(request.company_name)
        if not company_id:
            raise HTTPException(status_code=404, detail=f"Company not found: {request.company_name}")
        
        logger.info(f"Found company ID: {company_id} for {request.company_name}")
        
        # Get or create conversation context for this company
        conversation_context = request.conversation_context or []
        
        # Start Perplexity query immediately (parallel processing)
        logger.info("Starting Perplexity query in parallel")
        perplexity_task = asyncio.create_task(
            query_perplexity(request.query, request.company_name, conversation_context)
        )
        
        # Process company documents and analyze with Gemini
        logger.info("Starting document processing")
        processed_files = await process_company_documents(company_id, request.company_name)
        logger.info(f"Successfully processed {len(processed_files)} documents, now analyzing with Gemini")
        
        gemini_output = await analyze_documents_with_gemini(
            request.company_name, request.query, processed_files, conversation_context
        )
        logger.info("Completed Gemini document analysis")
        
        # Wait for Perplexity results
        logger.info("Waiting for Perplexity results to complete")
        perplexity_output, citations = await perplexity_task
        logger.info(f"Received Perplexity results with {len(citations)} citations")
        
        # Generate final answer with Claude
        logger.info("Starting Claude synthesis")
        claude_response = query_claude(
            request.query, request.company_name, gemini_output, perplexity_output, conversation_context
        )
        logger.info("Completed Claude synthesis")
        
        # Format sources section
        logger.info("Formatting response with sources")
        sources_section = "\n\n### Sources\n"
        
        # Prepare structured sources for the response
        structured_sources = {
            "company_data": [],
            "web_sources": []
        }
        
        # Add document sources under "Company data" sub-header
        sources_section += "\n#### Company data\n"
        for i, file_info in enumerate(processed_files, 1):
            # Get URL from file info
            if 'url' in file_info:
                url = file_info['url']
                # Remove trailing question mark if present
                if url.endswith('?'):
                    url = url[:-1]
                
                filename = os.path.basename(file_info['filename'])
                sources_section += f"{i}. [{filename}]({url})\n"
                
                # Add to structured sources
                structured_sources["company_data"].append({
                    "title": filename,
                    "url": url,
                    "type": file_info.get('type', 'document')
                })
        
        # Add Perplexity attribution and citations under "Web sources" sub-header
        sources_section += "\n#### Web sources\n"
        if citations:
            for i, citation in enumerate(citations, 1):
                # Handle different citation formats
                if isinstance(citation, str):
                    # If citation is just a URL string
                    url = citation
                    # Remove trailing question mark if present
                    if url.endswith('?'):
                        url = url[:-1]
                    
                    # Extract domain from URL if title is missing
                    try:
                        domain = urlparse(url).netloc
                        title = domain
                    except:
                        title = f"Source {i}"
                    sources_section += f"{i}. [{title}]({url})\n"
                    
                    # Add to structured sources
                    structured_sources["web_sources"].append({
                        "title": title,
                        "url": url
                    })
                elif isinstance(citation, dict):
                    # If citation is a dictionary object
                    url = citation.get("url", "")
                    # Remove trailing question mark if present
                    if url.endswith('?'):
                        url = url[:-1]
                    
                    title = citation.get("title", "")
                    if not title:
                        # Extract domain from URL if title is missing
                        try:
                            domain = urlparse(url).netloc
                            title = domain
                        except:
                            title = f"Source {i}"
                    sources_section += f"{i}. [{title}]({url})\n"
                    
                    # Add to structured sources
                    structured_sources["web_sources"].append({
                        "title": title,
                        "url": url
                    })
        else:
            sources_section += "*No specific web sources cited by Perplexity AI*"
        
        # Combine response with sources
        final_answer = claude_response + sources_section
        
        # Update conversation context with this exchange
        new_entry = {
            "query": request.query,
            "summary": claude_response[:500] + "..." if len(claude_response) > 500 else claude_response
        }
        conversation_context.append(new_entry)
        
        # Keep only the last 5 exchanges to prevent token overflow
        if len(conversation_context) > 5:
            conversation_context = conversation_context[-5:]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return {
            "answer": final_answer,
            "processing_time": processing_time,
            "sources": structured_sources
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {
        "greeting": "Hello, World!",
        "message": "Welcome to the Financial Insights API!"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"} 
