from dataclasses import dataclass
from typing import List
import json


@dataclass
class Product:
    """
    Data class representing a product for machine learning, containing only description, queries and images.
    """
    id: str
    generated_description: str
    compressed_jpg_urls: List[str]
    queries: List[str]
    
    @classmethod
    def from_db_row(cls, row: dict) -> 'Product':
        """
        Create a Product instance from a database row.
        
        Args:
            row: Dictionary containing product data from database
            
        Returns:
            Product instance
        """
        # Parse compressed_jpg_urls from JSON string if needed
        if isinstance(row['compressed_jpg_urls'], str):
            compressed_jpg_urls = json.loads(row['compressed_jpg_urls'])
        else:
            compressed_jpg_urls = row['compressed_jpg_urls']
            
        # Parse queries from JSON string if needed
        if isinstance(row['queries'], str):
            queries = json.loads(row['queries'])
        else:
            queries = row['queries']
        
        return cls(
            id=str(row['id']),
            generated_description=row['generated_description'],
            compressed_jpg_urls=compressed_jpg_urls,
            queries=queries
        )
