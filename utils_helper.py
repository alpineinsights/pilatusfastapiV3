"""
Helper file to resolve circular imports by re-exporting functions from app.py
"""

# This file will be imported by app.py to resolve the circular import issue
# The actual implementation is in app.py

# Function declarations that will be imported by app.py
def process_company_documents(*args, **kwargs):
    """
    Proxy function for process_company_documents in app.py
    This is a placeholder to avoid circular imports.
    The actual implementation is in app.py
    """
    # This will never be called, since app.py defines its own version
    pass

def initialize_claude(*args, **kwargs):
    """
    Proxy function for initialize_claude in app.py
    This is a placeholder to avoid circular imports.
    The actual implementation is in app.py
    """
    # This will never be called, since app.py defines its own version
    pass

import os
import aiohttp
import anthropic
import logging
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Helper function to initialize Claude client
def initialize_claude():
    """Initialize Claude client using environment variables"""
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

# Helper function to process company documents
async def process_company_documents(company_id: str, company_name: str, event_type: str = "all") -> List[Dict]:
    """
    Utility function to process company documents (imported by main.py)
    Returns a list of processed document information
    """
    # Import here to avoid circular imports
    from utils import QuartrAPI, AWSS3StorageHandler, TranscriptProcessor
    
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
            
            # Rest of the function will be imported from main.py to avoid duplication
            # This is just a stub to avoid circular imports
            return []
    except Exception as e:
        logger.error(f"Error in process_company_documents helper: {str(e)}")
        return [] 
