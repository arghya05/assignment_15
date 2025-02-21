import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from model import DeepSeekBlock
from config import DeepSeekConfig
from train import train

def prepare_data(config):
    # Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load smaller dataset for memory efficiency
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    
    def tokenize_function(examples):
        # Ensure text is a string
        text = examples["text"]
        if not text:
            text = " "  # Handle empty strings
            
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.sequence_length,
            return_tensors=None  # Changed from "pt" to None for batching
        )
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing texts"
    )
    
    # Convert to torch dataset
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Important for M2 memory management
    )
    
    return dataloader, tokenizer

def main():
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize config
    config = DeepSeekConfig()
    
    # Prepare data
    train_dataloader, tokenizer = prepare_data(config)
    
    # Initialize model
    model = DeepSeekBlock(config)
    
    # Initialize optimizer with lower learning rate for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Reduced learning rate
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=1000,  # Reduced number of epochs
        device=device
    )

if __name__ == "__main__":
    main() 