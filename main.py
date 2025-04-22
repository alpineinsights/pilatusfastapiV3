from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import asyncio
import aiohttp
import time
import logging
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from utils import QuartrAPI, AWSS3StorageHandler, TranscriptProcessor
from supabase_client import get_quartrid_by_name
from logger import logger
from urllib.parse import urlparse  # For parsing citation URLs

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
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Log the loaded API key (or lack thereof)
    if OPENROUTER_API_KEY:
        logger.info(f"OpenRouter API Key loaded successfully (length: {len(OPENROUTER_API_KEY)})")
    else:
        logger.error("OpenRouter API key NOT FOUND in environment variables")
        return None
    
    try:
        # Initialize the OpenAI client with OpenRouter base URL and API key
        # Following the official documentation pattern
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        logger.info("OpenAI client initialized for OpenRouter")
        return client
    except Exception as e:
        logger.error(f"Error initializing OpenRouter client: {str(e)}")
        return None

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
    """Call Perplexity API with a financial analyst prompt for the specified company using OpenRouter
    
    Args:
        query: The user's query
        company_name: The name of the company
        conversation_context: Passed explicitly to avoid thread issues with st.session_state
    
    Returns:
        Tuple[str, List[Dict]]: The response content and a list of citation objects
    """
    
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
        
        # Create system prompt for financial analysis instructions only
        system_prompt = f"""
        You are a helpful financial analyst assistant. The user is researching information about {company_name}.
        
        Make your answers as informative and well-structured as possible, organizing facts and figures in a clear and helpful way.
        Always cite your sources with URL references when possible. 
        Use only reliable sources, focusing on financial news, company reports, and expert analysis.
        {conversation_history}
        """
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Call Perplexity API via OpenRouter
        logger.info("OpenRouter Perplexity API: Sending request")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="perplexity/sonar-reasoning-pro",
            messages=messages,
            temperature=0.1,
            max_tokens=4000
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
            conversation_history = "\n\nPrevious conversation:\n"
            for entry in conversation_context:
                conversation_history += f"Question: {entry['query']}\n"
                conversation_history += f"Answer: {entry['summary']}\n\n"
                
        # Create system prompt
        system_prompt = f"""
        You are a senior financial analyst assistant. The user is asking about {company_name}.
        
        You have been provided with analysis from two sources:
        1. A document analysis that analyzed company-specific documents (financial reports, transcripts)
        2. A web search that searched for publicly available information
        
        Base your response primarily on this input data. If the sources contradict each other,
        favor the document analysis for company-specific facts.
        
        Make your answers informative and well-structured, organizing facts and figures in a clear
        and helpful way. Focus on factual information and provide helpful context about financial metrics.
        
        Present a balanced analysis without making specific investment recommendations.
        """
        
        # Format the input to the Claude API
        content = f"""
        USER QUERY: {query}
        
        GEMINI OUTPUT (Based on company documents):
        {gemini_output}
        
        PERPLEXITY OUTPUT (Web search):
        {perplexity_output}
        
        {conversation_history}
        """
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Call Claude API via OpenRouter
        logger.info("OpenRouter Claude API: Sending request")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            temperature=0.1,
            max_tokens=4000
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

# Function to analyze documents with Gemini
async def analyze_documents_with_gemini(company_name: str, query: str, processed_files: List[Dict], conversation_context=None):
    """Analyze company documents using Gemini AI via OpenRouter"""
    logger.info(f"Analyzing documents for {company_name} with OpenRouter Gemini")
    
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
        
        # Prepare documents for analysis
        documents_text = ""
        for doc in processed_files:
            documents_text += f"\n\n--- {doc['type'].upper()}: {doc['title']} ({doc['date']}) ---\n"
            if 'text' in doc:
                # For transcripts, we already have the text
                documents_text += doc['text']
            else:
                # For other documents, we just have the URL
                documents_text += f"Document URL: {doc['url']}\n"
        
        # Create prompt for Gemini
        prompt = f"""You are a financial analyst assistant specialized in analyzing company financial documents. 

Task: Please analyze the provided company documents for {company_name} and answer the following query: {query}

Base your analysis EXCLUSIVELY on the documents provided below. If the information isn't in the documents, state that clearly.
{conversation_history}

Here are the documents:
{documents_text}
"""

        # Create message structure
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call Gemini API via OpenRouter
        logger.info("OpenRouter Gemini API: Sending request")
        api_start_time = time.time()
        
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=messages,
            temperature=0.1,
            max_tokens=4000
        )
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenRouter Gemini API: Received response in {api_time:.2f} seconds")
        
        # Extract the text content from the response
        final_response = response.choices[0].message.content
        
        return final_response
    except Exception as e:
        logger.error(f"Error using OpenRouter Gemini API: {str(e)}")
        
        error_details = str(e)
        if "rate limit" in error_details.lower() or "429" in error_details:
            return "Error: Rate limit exceeded. Please try again in a moment."
        
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
        perplexity_task = asyncio.create_task(
            query_perplexity(request.query, request.company_name, conversation_context)
        )
        
        # Process company documents and analyze with Gemini
        processed_files = await process_company_documents(company_id, request.company_name)
        gemini_output = await analyze_documents_with_gemini(
            request.company_name, request.query, processed_files, conversation_context
        )
        
        # Wait for Perplexity results
        perplexity_output, citations = await perplexity_task
        
        # Generate final answer with Claude
        claude_response = query_claude(
            request.query, request.company_name, gemini_output, perplexity_output, conversation_context
        )
        
        # Format sources section
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
