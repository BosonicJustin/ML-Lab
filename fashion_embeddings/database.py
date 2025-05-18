from supabase import create_client
import supabase
from typing import List, Dict, Any


class SupabaseClient:
    """
    Singleton class to manage Supabase database connection.
    """
    _instance = None
    SCHEMA = "model_training"
    TABLE_NAME = "product_search_dataset_1"

    @classmethod
    def get_instance(cls, url: str = None, key: str = None):
        """
        Get or create singleton instance.
        
        Args:
            url: Supabase URL (only used when creating new instance)
            key: Supabase API key (only used when creating new instance)
        """
        if cls._instance is None:
            if url is None or key is None:
                raise ValueError("url and key must be provided when creating new instance")
            cls._instance = cls(url=url, key=key)
        return cls._instance

    def __init__(self, url: str, key: str):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase URL
            key: Supabase API key
        """
        if SupabaseClient._instance is not None:
            raise Exception("SupabaseClient is a singleton. Use get_instance() instead.")
            
        # Initialize Supabase client
        self.client = create_client(url, key)
        SupabaseClient._instance = self


    def fetch_product_ids(self) -> List[str]:
        """Fetch all product IDs from the database."""
        all_ids = []
        page_size = 1000
        start = 0
        
        while True:
            response = (
                self.client.schema(self.SCHEMA)
                .table(self.TABLE_NAME)
                .select("id")
                .range(start, start + page_size - 1)
                .execute()
            )
            
            if not response.data:
                break
                
            all_ids.extend(item['id'] for item in response.data)
            start += page_size
            
            if len(response.data) < page_size:
                break
                
        return all_ids
    
    def fetch_products_by_ids(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch products by their IDs.
        
        Args:
            product_ids: List of product IDs to fetch
            
        Returns:
            List of product dictionaries containing all fields
        """
        # Fetch in batches to avoid hitting API limits
        batch_size = 100
        all_products = []
        
        for i in range(0, len(product_ids), batch_size):
            batch_ids = product_ids[i:i + batch_size]
            response = (
                self.client.schema(self.SCHEMA)
                .table(self.TABLE_NAME)
                .select("*")
                .in_("id", batch_ids)
                .execute()
            )
            all_products.extend(response.data)
            
        return all_products


def main():
    """
    Initialize database client with credentials from config.
    """
    from config import get_supabase_credentials
    
    # Get credentials
    credentials = get_supabase_credentials()
    
    # Initialize client
    db = SupabaseClient.get_instance(
        url=credentials['url'],
        key=credentials['key']
    )

    product_ids = db.fetch_product_ids()

    products = db.fetch_products_by_ids(product_ids[100: 500])

    print(products)


if __name__ == "__main__":
    main()
