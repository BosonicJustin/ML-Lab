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
import pathlib

from database import SupabaseClient
from models import Product
from config import get_supabase_credentials

import torchvision

class ProductDataset(Dataset):
    """
    Dataset for loading product data from Supabase, can be used for both training and validation.
    """
    def __init__(self, product_ids: List[str], db: SupabaseClient, transform=None, mode="train", batch_size=32, cache_dir="image_cache", max_product_cache_size=1000):
        """
        Args:
            product_ids: List of product IDs to use
            db: SupabaseClient instance
            transform: Image transform to apply
            mode: Either "train" or "val". In train mode, only one random query is returned.
                 In val mode, all queries are returned.
            batch_size: Size of batches to load products in
            cache_dir: Directory to cache downloaded images
            max_product_cache_size: Maximum number of products to keep in memory cache
        """
        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"
        self.product_ids = product_ids
        self.db = db
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        
        # Create cache directory if it doesn't exist
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure the LRU cache size
        self._get_product_cached = lru_cache(maxsize=max_product_cache_size)(self._get_product_uncached)

    def _get_product_uncached(self, product_id: str) -> Product:
        """Get a product by ID from the database without caching."""
        
        # Fetch products
        products_data = self.db.fetch_products_by_ids([product_id])
        if not products_data:
            raise ValueError(f"Product {product_id} not found in database")
        
        return Product.from_db_row(products_data[0])
    
    def _get_product(self, product_id: str) -> Product:
        """Get a product by ID, using LRU cache."""
        return self._get_product_cached(product_id)
    
    def __len__(self):
        return len(self.product_ids)
    
    def _load_image(self, url: str) -> Optional[Image.Image]:
        """Load an image from URL with disk caching."""
        # Create filename from URL hash
        filename = str(hash(url)) + '.jpg'
        cache_path = self.cache_dir / filename
        
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception as e:
                print(f"Error loading cached image {cache_path}: {e}")
                cache_path.unlink()  # Delete corrupted cache file
                
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(cache_path)  # Cache the image
            return img
        except Exception as e:
            print(f"Error loading image {url}: {e}")
            return None
    
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


def main():
    """
    Initialize and test the ProductDataset.
    """
    from config import get_supabase_credentials
    from database import SupabaseClient
    import torch
    
    # Get credentials and initialize DB client
    credentials = get_supabase_credentials()
    db = SupabaseClient.get_instance(
        url=credentials['url'],
        key=credentials['key']
    )

    # Create dataset splits
    dataset_splits = ProductDatasetBuilder()
    
    # TODO: This is just a demo transform - for actual inference and training the transforms should be different
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # Get train and validation datasets
    train_dataset = dataset_splits.train(transform=transform)
    val_dataset = dataset_splits.val(transform=transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn
    )

    next(iter(train_loader))

    # print('START')
    # print(next(iter(train_loader)))
    # print('FINISH')

    # # Create dataloaders
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     collate_fn=collate_fn
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     collate_fn=collate_fn
    # )

    # # Print dataset information
    # train_size, val_size, test_size = dataset_splits.get_split_sizes()
    # print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # # Test a batch
    # for batch in train_loader:
    #     print("\nSample batch:")
    #     print(f"Images shape: {batch['images'].shape}")
    #     print(f"Image mask shape: {batch['image_mask'].shape}")
    #     print(f"Number of descriptions: {len(batch['descriptions'])}")
    #     print(f"Number of query lists: {len(batch['all_queries'])}")
    #     break


if __name__ == "__main__":
    main()
