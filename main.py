from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import asyncio
import aiohttp
import time
import logging
import json
import google.generativeai as genai
import anthropic
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

# Initialize Gemini model
def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not found in environment variables")
        return None
    
    try:
        # Configure the Gemini API with your API key
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error initializing Gemini: {str(e)}")
        return None

# Initialize Claude client
def initialize_claude():
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        logger.error("Claude API key not found in environment variables")
        return None
    
    try:
        # Initialize the Claude client
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Error initializing Claude: {str(e)}")
        return None

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
    """Call Perplexity API with a financial analyst prompt for the specified company"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
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
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Set timeout to prevent hanging requests
        timeout = aiohttp.ClientTimeout(total=45)  # 45 second timeout
        
        # Use aiohttp to make the request asynchronously
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
                
                # Extract citations if present
                citations = response_json.get("citations", [])
                logger.info(f"Perplexity API: Found {len(citations)} citations")
                
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
    """Analyze company documents using Gemini AI"""
    logger.info(f"Analyzing documents for {company_name} with Gemini")
    
    if not initialize_gemini():
        return "Error initializing Gemini AI"
    
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
        
        # Call Gemini API
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                generation_config={"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 4000}
            )
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as gemini_error:
            logger.error(f"Gemini API error: {str(gemini_error)}")
            return f"Error in Gemini analysis: {str(gemini_error)}"
    except Exception as e:
        logger.error(f"Error in document analysis: {str(e)}")
        return f"Error analyzing documents: {str(e)}"

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
