import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from product_representation import ProductEncoder, QueryEncoder
from dataset import ProductDatasetBuilder, collate_fn


def compute_all_embeddings(product_encoder, query_encoder, val_loader, device):
    """Compute embeddings for all products and their queries."""
    product_encoder.eval()
    query_encoder.eval()
    
    all_product_embeddings = []
    all_query_embeddings = []
    all_product_ids = []
    query_to_product_map = {}  # Maps query to its product_id
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing embeddings"):
            # Move data to device
            images = batch['images'].to(device)
            image_mask = batch['image_mask'].to(device)
            
            # Get product embeddings
            product_embeddings = product_encoder(
                images=images,
                description_texts=batch['descriptions'],
                image_mask=image_mask
            )
            
            # Process each product's queries
            for i, (product_id, queries) in enumerate(zip(batch['product_ids'], batch['all_queries'])):
                # Store product embedding
                all_product_embeddings.append(product_embeddings[i])
                all_product_ids.append(product_id)
                
                # Get query embeddings
                query_embeddings = query_encoder(queries)
                all_query_embeddings.append(query_embeddings)
                
                # Map each query to its product
                for query in queries:
                    query_to_product_map[query] = product_id
    
    # Concatenate all embeddings
    product_embeddings = torch.stack(all_product_embeddings)
    query_embeddings = torch.cat(all_query_embeddings)
    
    return product_embeddings, query_embeddings, all_product_ids, query_to_product_map


def evaluate_retrieval(product_embeddings, query_embeddings, product_ids, query_to_product_map, k_values=[1, 5, 10]):
    """
    Evaluate retrieval accuracy for different k values.
    Returns accuracy@k for each k in k_values.
    """
    # Normalize embeddings
    product_embeddings = F.normalize(product_embeddings, dim=1)
    query_embeddings = F.normalize(query_embeddings, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(query_embeddings, product_embeddings.t())
    
    # Get top-k indices for each query
    _, topk_indices = similarity.topk(max(k_values), dim=1)
    
    # Compute accuracy@k
    accuracies = {}
    for k in k_values:
        correct = 0
        total = 0
        
        for query_idx, (query, true_product_id) in enumerate(query_to_product_map.items()):
            topk_products = [product_ids[idx] for idx in topk_indices[query_idx][:k]]
            if true_product_id in topk_products:
                correct += 1
            total += 1
        
        accuracies[k] = correct / total
    
    return accuracies


def evaluate_model(product_encoder, query_encoder, val_loader, device, k_values=[1, 5, 10]):
    """
    Main evaluation function that takes a model and validation dataset and returns accuracy metrics.
    
    Args:
        product_encoder: The trained product encoder model
        query_encoder: The query encoder model
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        k_values: List of k values for accuracy@k computation
    
    Returns:
        dict: Dictionary of accuracy@k values
    """
    # Ensure models are in eval mode
    product_encoder.eval()
    query_encoder.eval()
    
    # Compute embeddings
    product_embeddings, query_embeddings, product_ids, query_to_product_map = compute_all_embeddings(
        product_encoder, query_encoder, val_loader, device
    )
    
    # Compute accuracy metrics
    return evaluate_retrieval(
        product_embeddings, query_embeddings,
        product_ids, query_to_product_map,
        k_values
    )


def main():
    """
    Example usage of the evaluation functions.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    product_encoder = ProductEncoder()
    query_encoder = QueryEncoder()
    
    # Load trained product encoder
    product_encoder.load_state_dict(torch.load('best_product_encoder.pth'))
    
    # Move models to device
    product_encoder = product_encoder.to(device)
    query_encoder = query_encoder.to(device)
    
    # Create dataset builder and get validation dataset
    dataset_builder = ProductDatasetBuilder(
        products_dir="./products",
        val_split=0.1,
        seed=42
    )
    
    # Print split sizes
    split_sizes = dataset_builder.get_split_sizes()
    print("Dataset splits:")
    for split, size in split_sizes.items():
        print(f"{split}: {size} products")
    print()
    
    # Create validation dataset and dataloader
    val_dataset = dataset_builder.val(transform=product_encoder.get_val_preprocess())
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Run evaluation
    print("Evaluating model...")
    accuracies = evaluate_model(product_encoder, query_encoder, val_loader, device)
    
    # Print results
    print("\nValidation Results:")
    for k, acc in accuracies.items():
        print(f"Accuracy@{k}: {acc:.4f}")


if __name__ == "__main__":
    main() 