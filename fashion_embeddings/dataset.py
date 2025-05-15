import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np


class ProductDataset(Dataset):
    """
    Dataset for loading product data, can be used for both training and validation.
    """
    def __init__(self, product_dirs, transform=None, mode="train"):
        """
        Args:
            product_dirs: List of product directory paths to use
            transform: Image transform to apply
            mode: Either "train" or "val". In train mode, only one random query is returned.
                 In val mode, all queries are returned.
        """
        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"
        self.product_dirs = product_dirs
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.product_dirs)
    
    def __getitem__(self, idx):
        product_dir = self.product_dirs[idx]
        product_id = os.path.basename(product_dir)
        
        # Load images
        image_paths = glob.glob(os.path.join(product_dir, "images", "*"))
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Load description
        desc_path = os.path.join(product_dir, "text", "description.txt")
        with open(desc_path, 'r', encoding='utf-8') as f:
            description = f.read().strip()
            
        # Load queries
        query_path = os.path.join(product_dir, "text", "query.txt")
        with open(query_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
            
        # For training, randomly select one query
        if self.mode == "train":
            queries = [np.random.choice(queries)]
            
        return {
            'product_id': product_id,
            'images': images,
            'description': description,
            'queries': queries,
            'num_images': len(images)
        }


class ProductDatasetBuilder:
    """
    Builder class that manages train/val splits and creates appropriate datasets.
    """
    def __init__(self, products_dir="./products", val_split=0.1, seed=42):
        """
        Args:
            products_dir: Root directory containing product folders
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducible splits
        """
        self.products_dir = products_dir
        
        # Get all product directories
        self.all_product_dirs = [d for d in glob.glob(os.path.join(products_dir, "*")) 
                                if os.path.isdir(d)]
        
        # Create reproducible splits
        np.random.seed(seed)
        indices = np.random.permutation(len(self.all_product_dirs))
        
        # Calculate split sizes
        n_total = len(self.all_product_dirs)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        # Create splits
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:]
        
        # Store split information
        self.splits = {
            'train': len(self.train_indices),
            'val': len(self.val_indices)
        }
        
    def train(self, transform=None):
        """Create training dataset"""
        train_dirs = [self.all_product_dirs[i] for i in self.train_indices]
        return ProductDataset(train_dirs, transform=transform, mode="train")
    
    def val(self, transform=None):
        """Create validation dataset"""
        val_dirs = [self.all_product_dirs[i] for i in self.val_indices]
        return ProductDataset(val_dirs, transform=transform, mode="val")
    
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
    
    # (B, Max, 3, 224, 224) - Product images 
    images = torch.zeros(batch_size, max_images, 3, 224, 224)
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
