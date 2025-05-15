import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from product_representation import ProductEncoder, QueryEncoder
from dataset import ProductDatasetBuilder, collate_fn
from evals import evaluate_model


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, product_embeddings, query_embeddings):
        # Normalize embeddings
        product_embeddings = nn.functional.normalize(product_embeddings, dim=1)
        query_embeddings = nn.functional.normalize(query_embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(query_embeddings, product_embeddings.t()) / self.temperature
        
        # Make sure for i'th image correspond to i'th query
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # -log(exp(...) / exp(sum ...))
        # In general make sure that the product corresponds to the query and is far away from other queries
        return nn.functional.cross_entropy(similarity, labels)


def train_model(
    product_encoder,
    query_encoder,
    train_loader,
    val_loader,
    device,
    num_epochs=20,
    learning_rate=1e-4,
    validate_every=2  # Validate every N epochs
):
    # Move models to device
    product_encoder = product_encoder.to(device)
    query_encoder = query_encoder.to(device)
    
    # Only optimize product encoder parameters (query encoder uses frozen SigLIP)
    optimizer = optim.AdamW(product_encoder.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        product_encoder.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move data to device
            images = batch['images'].to(device)
            image_mask = batch['image_mask'].to(device)
            
            # Get embeddings from both encoders
            product_embeddings = product_encoder(
                images=images,
                description_texts=batch['descriptions'],
                image_mask=image_mask
            )
            
            # In training mode, we only have one query per product
            queries = [q[0] for q in batch['all_queries']]  # Take first (only) query from each product
            query_embeddings = query_encoder(queries)
            
            # Compute loss
            loss = criterion(product_embeddings, query_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Validation phase
        if (epoch + 1) % validate_every == 0:
            print("\nRunning validation...")
            accuracies = evaluate_model(product_encoder, query_encoder, val_loader, device)
            
            # Print validation results
            print("\nValidation Results:")
            for k, acc in accuracies.items():
                print(f"Accuracy@{k}: {acc:.4f}")
            
            # Save best model
            if accuracies[1] > best_val_acc:  # Using accuracy@1 as the metric
                best_val_acc = accuracies[1]
                print(f"New best validation accuracy: {best_val_acc:.4f}")
                torch.save(product_encoder.state_dict(), 'best_product_encoder.pth')
            
            print()  # Empty line for readability


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    product_encoder = ProductEncoder()
    query_encoder = QueryEncoder()
    
    # Create dataset builder and get splits
    dataset_builder = ProductDatasetBuilder(
        products_dir="./products",
        val_split=1/3,
        seed=42
    )
    
    # Print split sizes
    split_sizes = dataset_builder.get_split_sizes()
    print("Dataset splits:")
    for split, size in split_sizes.items():
        print(f"{split}: {size} products")
    print()
    
    # Create datasets with appropriate preprocessing
    train_dataset = dataset_builder.train(transform=product_encoder.get_train_preprocess())
    val_dataset = dataset_builder.val(transform=product_encoder.get_val_preprocess())
    
    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,  # Can use larger batch size for validation
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Train model
    train_model(
        product_encoder=product_encoder,
        query_encoder=query_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10,
        validate_every=2  # Validate every 2 epochs
    )
    
    # Save final model
    torch.save(product_encoder.state_dict(), 'final_product_encoder.pth')


if __name__ == "__main__":
    main()
