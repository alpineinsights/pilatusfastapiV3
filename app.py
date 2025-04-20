import pandas as pd
import os
import tempfile
import uuid
import google.generativeai as genai
import time
import logging
from utils import QuartrAPI, AWSS3StorageHandler, TranscriptProcessor
import aiohttp
import asyncio
from typing import List, Dict, Tuple, Any, Optional
import json
import anthropic
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
from anthropic import Anthropic
from utils_helper import process_company_documents, initialize_claude
from datetime import datetime
from logging_config import setup_logging
from logger import logger  # Import the configured logger
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Financial Insights Chat",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_uploads" not in st.session_state:
    st.session_state.file_uploads = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "company_data" not in st.session_state:
    st.session_state.company_data = None
if "documents_fetched" not in st.session_state:
    st.session_state.documents_fetched = False
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

# Load credentials from Streamlit secrets - using flat structure
try:
    # Access API keys directly from root level
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    QUARTR_API_KEY = st.secrets["QUARTR_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
    
    # For debugging
    st.sidebar.write("Available secret keys:", list(st.secrets.keys()))
except KeyError as e:
    st.error(f"Missing required secret: {str(e)}. Please configure your secrets in Streamlit Cloud.")
    # Provide default values for development
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    QUARTR_API_KEY = os.environ.get("QUARTR_API_KEY", "")
    PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

# Load company data from Supabase
@st.cache_data(ttl=60*60)  # Cache for 1 hour
def load_company_data():
    """Load company data from Supabase"""
    companies = get_all_companies()
    if not companies:
        st.error("Failed to load company data from Supabase.")
        return None
    return pd.DataFrame(companies)

# Initialize Gemini model
def initialize_gemini():
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found in Streamlit secrets")
        return None
    
    try:
        # Configure the Gemini API with your API key
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

# Initialize Claude client
def initialize_claude():
    if not CLAUDE_API_KEY:
        st.error("Claude API key not found in Streamlit secrets")
        return None
    
    try:
        # Initialize the Claude client with only required parameters
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        return client
    except Exception as e:
        st.error(f"Error initializing Claude: {str(e)}")
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
    """Call Perplexity API with a financial analyst prompt for the specified company
    
    Args:
        query: The user's query
        company_name: The name of the company
        conversation_context: Passed explicitly to avoid thread issues with st.session_state
    
    Returns:
        Tuple[str, List[Dict]]: The response content and a list of citation objects
    """
    if not PERPLEXITY_API_KEY:
        logger.error("Perplexity API key not found")
        return "Error: Perplexity API key not found", []
    
    try:
        logger.info(f"Perplexity API: Starting request for query about {company_name}")
        start_time = time.time()
        
        url = "https://api.perplexity.ai/chat/completions"
        
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "Previous conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Create system prompt for financial analysis instructions only
        system_prompt = "You are a senior financial analyst on listed equities. Give comprehensive and detailed responses. Refrain from mentioning or making comments on stock price movements. Do not make any buy or sell recommendation."
        
        # Create user message with research context and the original query
        user_message = f"I am doing research on this listed company: {company_name}\n\n{query}\n\n{conversation_history}"
        
        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 2000,
            "temperature": 0.2,
            "web_search_options": {"search_context_size": "high"}
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Set timeout to prevent hanging requests
        timeout = aiohttp.ClientTimeout(total=45)  # 45 second timeout
        
        # Use aiohttp to make the request asynchronously
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info("Perplexity API: Sending request to API server")
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Perplexity API returned error {response.status}: {error_text}")
                        return f"Error: Perplexity API returned status {response.status}", []
                    
                    logger.info("Perplexity API: Received response from server")
                    response_json = await response.json()
                    elapsed = time.time() - start_time
                    logger.info(f"Perplexity API: Response received in {elapsed:.2f} seconds")
                    
                    # Log the raw response for debugging
                    logger.info(f"Perplexity API raw response structure: {list(response_json.keys())}")
                    
                    # Extract citations if present
                    citations = response_json.get("citations", [])
                    logger.info(f"Perplexity API: Found {len(citations)} citations")
                    # Log the format and content of the first few citations
                    if citations:
                        logger.info(f"Citation type: {type(citations)}")
                        for i, citation in enumerate(citations[:3]):  # Log first 3 citations
                            logger.info(f"Citation {i} type: {type(citation)}")
                            logger.info(f"Citation {i} content: {citation}")
                    else:
                        logger.info("No citations found in Perplexity response")
                    
                    # Simplified extraction - just get the text content directly
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        if "message" in response_json["choices"][0] and "content" in response_json["choices"][0]["message"]:
                            content = response_json["choices"][0]["message"]["content"]
                            
                            # If content contains a </think> tag, extract everything after it
                            if "</think>" in content:
                                content = content.split("</think>", 1)[1].strip()
                            
                            return content, citations
                    
                    # Fallback if we couldn't extract the content using the above method
                    try:
                        parsed_content = extract_valid_json(response_json)
                        if isinstance(parsed_content, dict) and "content" in parsed_content:
                            return parsed_content["content"], citations
                        return str(parsed_content), citations
                    except Exception as e:
                        logger.error(f"Error parsing Perplexity response: {e}")
                        return "Error processing Perplexity response. Please check logs for details.", []
        except asyncio.TimeoutError:
            logger.error("Perplexity API request timed out after 45 seconds")
            return "Error: Perplexity API request timed out. Please try again later.", []
        except asyncio.CancelledError:
            logger.info("Perplexity API request was cancelled")
            raise  # Re-raise the CancelledError to allow proper cleanup
        
    except asyncio.CancelledError:
        logger.warning("Perplexity API task was cancelled")
        raise  # Re-raise to allow proper cleanup
    except Exception as e:
        logger.error(f"Error calling Perplexity API: {str(e)}")
        return f"Error calling Perplexity API: {str(e)}", []

# Function to call Claude with combined outputs
def query_claude(query: str, company_name: str, gemini_output: str, perplexity_output: str, conversation_context=None) -> str:
    """Call Claude API with combined Gemini and Perplexity outputs for final synthesis"""
    logger.info("Claude API: Starting synthesis process")
    start_time = time.time()
    
    client = initialize_claude()
    if not client:
        return "Error initializing Claude client"
    
    try:
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Create prompt for Claude
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
        
        logger.info(f"Claude API: Sending request with prompt length {len(prompt)} characters")
        api_start_time = time.time()
        
        # Call Claude API with the updated model name
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            system="You are a senior financial analyst providing detailed analysis for professional investors.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        api_time = time.time() - api_start_time
        logger.info(f"Claude API: Received response in {api_time:.2f} seconds")
        
        response_text = message.content[0].text
        total_time = time.time() - start_time
        logger.info(f"Claude API: Total processing time: {total_time:.2f} seconds")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return f"Error calling Claude API: {str(e)}"

# Function to process company documents
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
            
            # Use the company name passed in directly from Supabase
            logger.info(f"Processing documents for company: {company_name} (ID: {company_id})")
                
            events = company_data.get('events', [])
            if not events:
                logger.warning(f"No events found for company: {company_name} (ID: {company_id})")
                return []
                
            # Sort events by date (descending) - should already be sorted, but just to be sure
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
                    
                event_date = event.get('eventDate', '').split('T')[0]
                event_title = event.get('eventTitle', 'Unknown Event')
                
                # Process PDF/slides (if we need more)
                if pdf_count < 2 and event.get('pdfUrl'):
                    try:
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
                                    content, filename, 
                                    response.headers.get('content-type', 'application/pdf')
                                )
                                
                                if success:
                                    public_url = storage_handler.get_public_url(filename)
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'slides',
                                        'event_date': event_date,
                                        'event_title': event_title,
                                        'url': public_url,
                                        'storage_type': 'supabase'
                                    })
                                    pdf_count += 1
                    except Exception as e:
                        st.error(f"Error processing slides for {event_title}: {str(e)}")
                
                # Process report (if we need more)
                if report_count < 2 and event.get('reportUrl'):
                    try:
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
                                    content, filename, 
                                    response.headers.get('content-type', 'application/pdf')
                                )
                                
                                if success:
                                    public_url = storage_handler.get_public_url(filename)
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'report',
                                        'event_date': event_date,
                                        'event_title': event_title,
                                        'url': public_url,
                                        'storage_type': 'supabase'
                                    })
                                    report_count += 1
                    except Exception as e:
                        st.error(f"Error processing report for {event_title}: {str(e)}")
                
                # Only process the transcript if we need more
                if transcript_count < 2 and event.get('transcriptUrl'):
                    # Process transcript
                    try:
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
                                    'event_date': event_date,
                                    'event_title': event_title,
                                    'url': public_url,
                                    'storage_type': 'supabase'
                                })
                                transcript_count += 1
                    except Exception as e:
                        st.error(f"Error processing transcript for {event_title}: {str(e)}")
            
            # Log the number of documents processed
            logger.info(f"Processed {pdf_count} PDFs, {report_count} reports, and {transcript_count} transcripts")
            return processed_files
    except Exception as e:
        st.error(f"Error processing company documents: {str(e)}")
        return []

# Function to download files from storage to temporary location
async def download_files_from_s3(file_urls: List[str]) -> List[str]:
    """Download files from AWS S3 storage to temporary location and return local paths"""
    try:
        aws_handler = AWSS3StorageHandler()
        temp_dir = tempfile.mkdtemp()
        local_files = []
        
        # Create the asyncio event loop if not already running
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
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
                success = loop.run_until_complete(aws_handler.download_file(s3_key, local_path))
                
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
    """Query Gemini model with context from files"""
    try:
        logger.info(f"Gemini API: Starting analysis with {len(file_paths)} documents")
        start_time = time.time()
        
        # Make sure Gemini is initialized
        if not initialize_gemini():
            return "Error initializing Gemini client"
        
        # Create a model instance with proper configuration
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=7000
            )
        )
        
        # Build conversation history for context (safely)
        conversation_history = ""
        if conversation_context:
            conversation_history = "Previous conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
        
        # Prepare files and content parts
        contents = []
        
        # Add files to contents
        logger.info("Gemini API: Processing document files")
        file_start_time = time.time()
        for file_path in file_paths:
            try:
                # Open the file and create a content part
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    
                # Add file data as content part
                file_mime_type = 'application/pdf'  # Assuming all files are PDFs
                contents.append({
                    'mime_type': file_mime_type,
                    'data': file_data
                })
            except Exception as e:
                st.error(f"Error processing file for Gemini: {str(e)}")
        
        file_processing_time = time.time() - file_start_time
        logger.info(f"Gemini API: Processed {len(contents)} files in {file_processing_time:.2f} seconds")
        
        if not contents:
            return "No files were successfully processed for Gemini"
        
        # Add the prompt as the final content
        prompt = f"You are a senior financial analyst. Review the attached documents and provide a detailed and structured answer to the user's query. User's query: '{query}'\n\n{conversation_history}"
        contents.append(prompt)
        
        # Generate content with files as context
        logger.info("Gemini API: Sending request to API")
        api_start_time = time.time()
        response = model.generate_content(contents)
        api_time = time.time() - api_start_time
        logger.info(f"Gemini API: Received response in {api_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Gemini API: Total processing time: {total_time:.2f} seconds")
        
        # Return the response text
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

# Main UI components
def main():
    st.title("Financial Insights Chat")
    
    # Load company data
    company_data = load_company_data()
    if company_data is None:
        st.error("Failed to load company data. Please check the Supabase connection.")
        return
    
    # Sidebar with company selection
    with st.sidebar:
        st.header("Select Company")
        company_names = get_company_names()
        selected_company = st.selectbox(
            "Choose a company:",
            options=company_names,
            index=0 if company_names else None
        )
        
        if selected_company:
            # Get both Quartr ID (primary) and ISIN (legacy)
            quartr_id = get_quartrid_by_name(selected_company)
            isin = get_isin_by_name(selected_company)
            
            # Display Quartr ID for debugging
            st.info(f"Quartr ID: {quartr_id}")
            
            # Check if company changed
            if st.session_state.current_company != selected_company:
                st.session_state.current_company = selected_company
                st.session_state.company_data = {
                    'name': selected_company,
                    'isin': isin,  # Keep for backward compatibility
                    'quartr_id': quartr_id  # Primary identifier
                }
                
                # Clear previous conversation when company changes
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.session_state.documents_fetched = False
                st.session_state.conversation_context = []
        
        # Add information about conversation capabilities
        st.markdown("---")
        st.markdown("### Conversation Features")
        st.info("This app now supports follow-up questions! The AI will remember previous exchanges and provide contextual responses.")
    
    # Main chat area
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Main chat input section
    # Chat input with updated placeholder
    if query := st.chat_input("Ask about the company or follow up on previous answers..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            # Check if we have a selected company
            if not st.session_state.company_data:
                response = "Please select a company from the sidebar first."
                response_placeholder.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                return
            
            # Get company name for Perplexity
            company_name = st.session_state.company_data['name']
            
            # Get conversation context safely before starting thread
            conversation_context = list(st.session_state.conversation_context) if "conversation_context" in st.session_state else []
            
            # Create an event loop in a new thread for Perplexity
            perplexity_loop = asyncio.new_event_loop()
            
            # Define a function to run the Perplexity query in a separate thread
            def run_perplexity_query():
                asyncio.set_event_loop(perplexity_loop)
                perplexity_loop.run_forever()
            
            # Start the perplexity thread
            perplexity_thread = threading.Thread(target=run_perplexity_query, daemon=True)
            perplexity_thread.start()
            
            # Start Perplexity API call immediately
            logger.info(f"Starting Perplexity API call immediately for query about {company_name}")
            start_time = time.time()
            perplexity_future = asyncio.run_coroutine_threadsafe(query_perplexity(query, company_name, conversation_context), perplexity_loop)
            
            try:
                # Fetch documents if not already fetched
                if not st.session_state.documents_fetched:
                    with st.spinner(f"Fetching documents for {st.session_state.company_data['name']}..."):
                        # Use Quartr ID to fetch documents instead of ISIN
                        quartr_id = st.session_state.company_data['quartr_id']
                        if not quartr_id:
                            response = "No Quartr ID found for this company. Please select another company."
                            response_placeholder.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            # Clean up
                            perplexity_future.cancel()
                            perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                            perplexity_thread.join(timeout=1.0)
                            return
                            
                        # Process company documents using the Quartr ID
                        processed_files = asyncio.run(process_company_documents(quartr_id, st.session_state.company_data['name']))
                        st.session_state.processed_files = processed_files
                        st.session_state.documents_fetched = True
                        
                        if not processed_files:
                            response = "No documents found for this company. Please try another company or check your Quartr API key."
                            response_placeholder.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            # Clean up
                            perplexity_future.cancel()
                            perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                            perplexity_thread.join(timeout=1.0)
                            return
                
                # Process the user query with fetched documents (for both new and follow-up questions)
                if st.session_state.processed_files:
                    with st.spinner("Processing your query with multiple AI models..."):
                        # Download files from storage
                        local_files = asyncio.run(download_files_from_s3([file_info['url'] for file_info in st.session_state.processed_files]))
                        
                        if not local_files:
                            response = "Error downloading files from storage. Please check your connection."
                            response_placeholder.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            # Clean up
                            perplexity_future.cancel()
                            perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                            perplexity_thread.join(timeout=1.0)
                            return
                        
                        # Run Gemini analysis on documents
                        logger.info("Starting Gemini analysis on documents")
                        gemini_start = time.time()
                        gemini_output = query_gemini(query, local_files, conversation_context)
                        gemini_duration = time.time() - gemini_start
                        logger.info(f"Completed Gemini analysis in {gemini_duration:.2f} seconds")
                        
                        # Wait for Perplexity to complete (it's been running since the beginning)
                        logger.info("Waiting for Perplexity to complete (if not already finished)")
                        try:
                            # Get result with timeout
                            perplexity_output, perplexity_citations = perplexity_future.result(timeout=60)
                            perplexity_duration = time.time() - start_time
                            logger.info(f"Completed Perplexity request in {perplexity_duration:.2f} seconds")
                            logger.info(f"Perplexity returned {len(perplexity_citations)} citations")
                        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception) as e:
                            # Handle timeout, cancellation, or other errors
                            logger.error(f"Error waiting for Perplexity task: {str(e)}")
                            # Cancel the task if it's still running
                            if not perplexity_future.done():
                                logger.info("Cancelling Perplexity task as it's still running")
                                perplexity_future.cancel()
                            perplexity_output = "Error: Perplexity API request timed out or failed."
                            perplexity_citations = []
                        
                        # Clean up the perplexity thread and loop
                        try:
                            perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                            perplexity_thread.join(timeout=1.0)
                            logger.info("Successfully cleaned up Perplexity thread")
                        except Exception as e:
                            logger.error(f"Error cleaning up Perplexity thread: {str(e)}")
                        
                        # Log completion 
                        logger.info("Completed first-stage LLM processing (Gemini and Perplexity)")
                        logger.info(f"Gemini output length: {len(gemini_output)} characters")
                        logger.info(f"Perplexity output length: {len(perplexity_output)} characters")
                        
                        # Error handling
                        if gemini_output.startswith("Error") and perplexity_output.startswith("Error"):
                            response = "Both Gemini and Perplexity APIs failed. Please try again later."
                            response_placeholder.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            return
                        
                        # Process with Claude
                        logger.info("Starting final synthesis with Claude")
                        claude_start = time.time()
                        claude_response = query_claude(query, company_name, gemini_output, perplexity_output, conversation_context)
                        claude_duration = time.time() - claude_start
                        logger.info(f"Completed Claude synthesis in {claude_duration:.2f} seconds")
                        
                        # Format sources section
                        sources_section = "\n\n### Sources\n"
                        
                        # Add document sources under "Company data" sub-header
                        sources_section += "\n#### Company data\n"
                        for i, file_info in enumerate(st.session_state.processed_files, 1):
                            # Get URL from file info
                            if 'url' in file_info:
                                url = file_info['url']
                                filename = os.path.basename(file_info['filename'])
                                sources_section += f"{i}. [{filename}]({url})\n"
                        
                        # Add Perplexity attribution and citations under "Web sources" sub-header
                        sources_section += "\n#### Web sources\n"
                        if perplexity_citations:
                            for i, citation in enumerate(perplexity_citations, 1):
                                # Handle different citation formats
                                if isinstance(citation, str):
                                    # If citation is just a URL string
                                    url = citation
                                    # Extract domain from URL if title is missing
                                    try:
                                        from urllib.parse import urlparse
                                        domain = urlparse(url).netloc
                                        title = domain
                                    except:
                                        title = f"Source {i}"
                                    sources_section += f"{i}. [{title}]({url})\n"
                                elif isinstance(citation, dict):
                                    # If citation is a dictionary object
                                    url = citation.get("url", "")
                                    title = citation.get("title", "")
                                    if not title:
                                        # Extract domain from URL if title is missing
                                        try:
                                            from urllib.parse import urlparse
                                            domain = urlparse(url).netloc
                                            title = domain
                                        except:
                                            title = f"Source {i}"
                                    sources_section += f"{i}. [{title}]({url})\n"
                        else:
                            sources_section += "*No specific web sources cited by Perplexity AI*"
                        
                        # Combine response with sources
                        final_response = claude_response + sources_section
                        
                        # Calculate total processing time
                        total_duration = time.time() - start_time
                        logger.info(f"Total processing time: {total_duration:.2f} seconds")
                        
                        # Display response with sources
                        response_placeholder.markdown(final_response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                        
                        # Save condensed version of the response to conversation context
                        # Extract the response without the sources section
                        response_without_sources = claude_response.split("\n\n### Sources")[0]
                        
                        # Add to conversation context for future reference
                        st.session_state.conversation_context.append({
                            "query": query,
                            "summary": response_without_sources[:500] + "..." if len(response_without_sources) > 500 else response_without_sources
                        })
                        
                        # Limit conversation context to last 5 exchanges to prevent token overflow
                        if len(st.session_state.conversation_context) > 5:
                            st.session_state.conversation_context = st.session_state.conversation_context[-5:]
                else:
                    response = "No documents are available for this company. Please try another company."
                    response_placeholder.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    # Clean up
                    perplexity_future.cancel()
                    perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                    perplexity_thread.join(timeout=1.0)
            except Exception as e:
                # Handle any unexpected errors
                logger.error(f"Unexpected error during processing: {str(e)}")
                response = f"An unexpected error occurred: {str(e)}"
                response_placeholder.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                # Clean up
                perplexity_future.cancel()
                perplexity_loop.call_soon_threadsafe(perplexity_loop.stop)
                perplexity_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
