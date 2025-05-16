import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
import requests
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Set
import random
from functools import lru_cache

from database import SupabaseClient
from models import Product
from config import get_supabase_credentials


class ProductDataset(Dataset):
    """
    Dataset for loading product data from Supabase, can be used for both training and validation.
    """
    def __init__(self, product_ids: List[str], db: SupabaseClient, transform=None, mode="train", batch_size=32):
        """
        Args:
            product_ids: List of product IDs to use
            db: SupabaseClient instance
            transform: Image transform to apply
            mode: Either "train" or "val". In train mode, only one random query is returned.
                 In val mode, all queries are returned.
            batch_size: Size of batches to load products in
        """
        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"
        self.product_ids = product_ids
        self.db = db
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        
        # Cache for loaded images
        self._image_cache: Dict[str, Image.Image] = {}
        # Cache for loaded products
        self._product_cache: Dict[str, Product] = {}
        
    def __len__(self):
        return len(self.product_ids)
    
    def _load_image(self, url: str) -> Optional[Image.Image]:
        """Load an image from URL with caching."""
        if url in self._image_cache:
            return self._image_cache[url]
            
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            self._image_cache[url] = img
            return img
        except Exception as e:
            print(f"Error loading image {url}: {e}")
            return None
    
    def _get_product(self, product_id: str) -> Product:
        """Get a product by ID, loading from database if necessary."""
        if product_id not in self._product_cache:
            # Load the batch of products containing this ID
            batch_start = (self.product_ids.index(product_id) // self.batch_size) * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(self.product_ids))
            batch_ids = self.product_ids[batch_start:batch_end]
            
            # Fetch only products not in cache
            missing_ids = [pid for pid in batch_ids if pid not in self._product_cache]
            if missing_ids:
                products_data = self.db.fetch_products_by_ids(missing_ids)
                for row in products_data:
                    product = Product.from_db_row(row)
                    self._product_cache[str(product.id)] = product
        
        return self._product_cache[product_id]
    
    def __getitem__(self, idx):
        product_id = self.product_ids[idx]
        product = self._get_product(product_id)
        
        # Load images
        images = []
        for url in product.compressed_jpg_urls:
            img = self._load_image(url)
            if img is not None and self.transform:
                img = self.transform(img)
                images.append(img)
        
        # Ensure we have at least one image
        if not images:
            raise ValueError(f"No valid images found for product {product.id}")
            
        # Stack images into tensor
        images = torch.stack(images)
        
        # For training, randomly select one query
        queries = product.queries
        if self.mode == "train":
            queries = [random.choice(queries)]
            
        return {
            'product_id': str(product.id),
            'images': images,
            'description': product.generated_description,
            'queries': queries,
            'num_images': len(images)
        }


class ProductDatasetBuilder:
    """
    Builder class that manages train/val splits and creates appropriate datasets.
    """
    def __init__(self, val_split=0.1, seed=42, batch_size=32):
        """
        Args:
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducible splits
            batch_size: Size of batches to load products in
        """
        # Get Supabase credentials and initialize client
        credentials = get_supabase_credentials()
        self.db = SupabaseClient.get_instance(url=credentials['url'], key=credentials['key'])
        self.batch_size = batch_size
        
        # Fetch all product IDs
        self.all_product_ids = self.db.fetch_product_ids()
    
        # Create reproducible splits
        np.random.seed(seed)
        indices = np.random.permutation(len(self.all_product_ids))
        
        # Calculate split sizes
        n_total = len(self.all_product_ids)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        # Create splits
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Store product IDs for each split
        self.train_product_ids = [self.all_product_ids[i] for i in train_indices]
        self.val_product_ids = [self.all_product_ids[i] for i in val_indices]
        
        # Store split information
        self.splits = {
            'train': len(self.train_product_ids),
            'val': len(self.val_product_ids)
        }
        
    def train(self, transform=None):
        """Create training dataset"""
        return ProductDataset(
            self.train_product_ids,
            self.db,
            transform=transform,
            mode="train",
            batch_size=self.batch_size
        )
    
    def val(self, transform=None):
        """Create validation dataset"""
        return ProductDataset(
            self.val_product_ids,
            self.db,
            transform=transform,
            mode="val",
            batch_size=self.batch_size
        )
    
    def get_split_sizes(self):
        """Return the sizes of each split"""
        return self.splits


def collate_fn(batch):
    """
    Collate function that handles variable number of images per product.
    Works for both training and validation.
    """
    max_images = max(item['num_images'] for item in batch)
    batch_size = len(batch)
    
    # Get shape from first image
    C, H, W = batch[0]['images'][0].shape
    
    # (B, Max, C, H, W) - Product images 
    images = torch.zeros(batch_size, max_images, C, H, W)
    image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)
    descriptions = []
    product_ids = []
    all_queries = []  # List of lists of queries (single query for train, multiple for val)
    
    for i, item in enumerate(batch):
        for j, img in enumerate(item['images']):
            images[i, j] = img
            image_mask[i, j] = True
        descriptions.append(item['description'])
        product_ids.append(item['product_id'])
        all_queries.append(item['queries'])
    
    return {
        'product_ids': product_ids,
        'images': images,
        'image_mask': image_mask,
        'descriptions': descriptions,
        'all_queries': all_queries
    }
