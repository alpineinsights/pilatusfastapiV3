"""
This module handles the integration with Supabase to fetch company data.
Uses the Supabase Python client for more reliable connections.
"""

import os
from supabase import create_client
from typing import Dict, List, Optional
import pandas as pd
import logging
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Supabase client
@lru_cache(maxsize=1)
def init_client():
    """
    Initialize and cache the Supabase client connection
    """
    try:
        # Use the hardcoded credentials as fallback
        supabase_url = "https://maeistbokyjhewrrisvf.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1hZWlzdGJva3lqaGV3cnJpc3ZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMxNTgyMTUsImV4cCI6MjA1ODczNDIxNX0._Fb4I1BvmqMHbB5KyrtlEmPTyF8nRgR9LsmNFmiZSN8"
        
        # Override with values from environment variables if available
        env_url = os.getenv("SUPABASE_URL")
        env_key = os.getenv("SUPABASE_ANON_KEY")
        
        if env_url:
            supabase_url = env_url
        if env_key:
            supabase_key = env_key
                
        # Initialize the client
        client = create_client(supabase_url, supabase_key)
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        return None

@lru_cache(maxsize=100)
def get_all_companies() -> List[Dict]:
    """
    Fetches all companies from the Supabase 'universe' table.
    
    Returns:
        List[Dict]: A list of company data dictionaries
    """
    try:
        client = init_client()
        if not client:
            return []
            
        response = client.table('universe').select('*').execute()
        if hasattr(response, 'data'):
            return response.data
        return []
    except Exception as e:
        logger.error(f"Error fetching companies from Supabase: {str(e)}")
        return []

@lru_cache(maxsize=100)
def get_company_names() -> List[str]:
    """
    Returns a list of all company names from Supabase.
    
    Returns:
        List[str]: A list of company names
    """
    companies = get_all_companies()
    return [company["Name"] for company in companies if "Name" in company]

@lru_cache(maxsize=100)
def get_quartrid_by_name(company_name: str) -> Optional[str]:
    """
    Retrieves the Quartr ID for a given company name from Supabase.
    
    Args:
        company_name (str): The company name to look up
        
    Returns:
        str: The Quartr ID if found, None otherwise
    """
    try:
        client = init_client()
        if not client:
            return None
            
        response = client.table('universe').select('\"Quartr Id\"').eq('Name', company_name).execute()
        if response.data and len(response.data) > 0:
            quartr_id = response.data[0].get("Quartr Id")
            logger.info(f"Found Quartr ID {quartr_id} for company: {company_name}")
            return str(quartr_id)  # Convert to string to ensure compatibility
        return None
    except Exception as e:
        logger.error(f"Error fetching Quartr ID for {company_name}: {str(e)}")
        return None

@lru_cache(maxsize=100)
def get_company_by_quartrid(quartrid: str) -> Optional[Dict]:
    """
    Retrieves company data for a given Quartr ID from Supabase.
    
    Args:
        quartrid (str): The Quartr ID to look up
        
    Returns:
        dict: The company data if found, None otherwise
    """
    try:
        client = init_client()
        if not client:
            return None
            
        response = client.table('universe').select('*').eq('\"Quartr Id\"', quartrid).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching company by Quartr ID {quartrid}: {str(e)}")
        return None
