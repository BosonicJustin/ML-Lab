from supabase import create_client
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
        """
        Fetch all product IDs from the database.
        
        Returns:
            List of product IDs as strings
        """
        try:
            # First try to get the count
            count = self.get_product_count()
            print(f"Total products in table: {count}")
            
            # Try a basic select query
            response = self.client.schema(self.SCHEMA) \
                .from_(self.TABLE_NAME) \
                .select('id') \
                .execute()
            
            print("Response metadata:", {
                "status_code": getattr(response, 'status_code', None),
                "data_length": len(response.data) if response.data else 0,
                "has_error": hasattr(response, 'error') and response.error is not None
            })
            
            if hasattr(response, 'error') and response.error:
                print("Query error:", response.error)
                return []
                
            if not response.data:
                print("No data returned. Trying sample query...")
                # Try selecting all columns with limit 1 to see the structure
                sample_response = self.client.schema(self.SCHEMA) \
                    .from_(self.TABLE_NAME) \
                    .select('*') \
                    .limit(1) \
                    .execute()
                    
                print("Sample query response:", {
                    "has_data": bool(sample_response.data),
                    "columns": list(sample_response.data[0].keys()) if sample_response.data else None
                })
                
            return [str(row['id']) for row in response.data] if response.data else []
            
        except Exception as e:
            print(f"Error fetching product IDs: {str(e)}")
            return []
    
    def fetch_products_by_ids(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch specific products by their IDs.
        
        Args:
            product_ids: List of product IDs to fetch
            
        Returns:
            List of product dictionaries
        """
        try:
            response = self.client.schema(self.SCHEMA) \
                .from_(self.TABLE_NAME) \
                .select('*') \
                .in_('id', product_ids) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                print("Query error:", response.error)
                return []
                
            return response.data if response.data else []
            
        except Exception as e:
            print(f"Error fetching products by IDs: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """
        Test database connection and permissions.
        
        Returns:
            bool: True if connection and permissions are valid
        """
        try:
            # Test basic connection
            print("Testing database connection...")
            
            # Test schema access
            schema_response = self.client.rpc('get_schema_name').execute()
            print("Current schema:", schema_response.data)
            
            # Test table access
            table_response = self.client.schema(self.SCHEMA) \
                .from_(self.TABLE_NAME) \
                .select('count(*)', count='exact') \
                .limit(1) \
                .execute()
            
            print("Table access test:", {
                "schema": self.SCHEMA,
                "table": self.TABLE_NAME,
                "success": bool(table_response.data),
                "row_count": table_response.count if hasattr(table_response, 'count') else None
            })
            
            # Test RLS policies
            policy_response = self.client.rpc('get_policies', {
                'table_name': f"{self.SCHEMA}.{self.TABLE_NAME}"
            }).execute()
            
            print("RLS policies:", policy_response.data if hasattr(policy_response, 'data') else None)
            
            return True
            
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False

    def verify_table_structure(self) -> bool:
        """
        Verify that the table has the expected structure.
        
        Returns:
            bool: True if table structure matches expected schema
        """
        try:
            # Get table information
            table_info = self.client.rpc('get_table_info', {
                'table_name': f"{self.SCHEMA}.{self.TABLE_NAME}"
            }).execute()
            
            expected_columns = {
                'id': 'uuid',
                'generated_description': 'text',
                'compressed_jpg_urls': 'text[]',
                'brand_id': 'uuid',
                'title': 'text',
                'url': 'text',
                'category': 'text',
                'style': 'text',
                'queries': 'text[]',
                'created_at': 'timestamp'
            }
            
            if not table_info.data:
                print("Could not retrieve table information")
                return False
                
            actual_columns = {col['column_name']: col['data_type'] for col in table_info.data}
            
            # Compare expected vs actual
            missing_columns = set(expected_columns.keys()) - set(actual_columns.keys())
            extra_columns = set(actual_columns.keys()) - set(expected_columns.keys())
            mismatched_types = {
                col: (expected_columns[col], actual_columns[col])
                for col in set(expected_columns.keys()) & set(actual_columns.keys())
                if expected_columns[col] != actual_columns[col]
            }
            
            print("Table structure verification:", {
                "missing_columns": list(missing_columns) if missing_columns else None,
                "extra_columns": list(extra_columns) if extra_columns else None,
                "mismatched_types": mismatched_types if mismatched_types else None,
                "is_valid": not (missing_columns or mismatched_types)
            })
            
            return not (missing_columns or mismatched_types)
            
        except Exception as e:
            print(f"Table structure verification failed: {str(e)}")
            return False
