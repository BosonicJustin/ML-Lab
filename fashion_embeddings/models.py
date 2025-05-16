from dataclasses import dataclass
from typing import List
from datetime import datetime
import uuid
import json


@dataclass
class Product:
    """
    Data class representing a product from the database.
    """
    id: uuid.UUID
    generated_description: str
    compressed_jpg_urls: List[str]
    brand_id: uuid.UUID
    title: str
    url: str
    category: str
    style: str
    queries: List[str]
    created_at: datetime
    
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
            
        # Parse UUIDs
        id_ = uuid.UUID(row['id']) if isinstance(row['id'], str) else row['id']
        brand_id = uuid.UUID(row['brand_id']) if isinstance(row['brand_id'], str) else row['brand_id']
        
        # Parse timestamp
        created_at = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')) \
            if isinstance(row['created_at'], str) else row['created_at']
        
        return cls(
            id=id_,
            generated_description=row['generated_description'],
            compressed_jpg_urls=compressed_jpg_urls,
            brand_id=brand_id,
            title=row['title'],
            url=row['url'],
            category=row['category'],
            style=row['style'],
            queries=queries,
            created_at=created_at
        ) 