import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_supabase_credentials():
    """
    Get Supabase credentials from environment variables.
    Raises ValueError if required credentials are not set.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
    return {
        "url": supabase_url,
        "key": supabase_key
    } 