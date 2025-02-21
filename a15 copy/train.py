import torch
import gc
from tqdm import tqdm
from datetime import datetime
import os
import time

def save_training_log(log_content, filename="training_log.txt"):
    with open(filename, "a") as f:
        f.write(log_content + "\n")

def train(model, train_dataloader, optimizer, num_epochs=1000, device='mps'):
    # Create log directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.txt"
    
    # Initial log entry
    initial_log = f"""Training Log for DeepSeek Model (M2 Mac Optimized)
Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Training Parameters:
- Batch Size: {train_dataloader.batch_size}
- Gradient Accumulation Steps: 8
- Device: {device}
- Number of Epochs: {num_epochs}

Training Progress:
"""
    save_training_log(initial_log, log_file)
    
    # Memory optimization for M2
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    
    gradient_accumulation_steps = 8
    start_time = time.time()  # Add start time for speed calculation
    
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        last_loss = 0  # Track the last loss value
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx % 20 == 0:
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs, load_balancing_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = outputs.mean() + 0.01 * load_balancing_loss
            loss = loss / gradient_accumulation_steps
            last_loss = loss.item()  # Save the last loss value
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += last_loss * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': f'{last_loss:.4f}'})
            
            del input_ids, attention_mask, outputs, loss
            
        avg_loss = total_loss / len(train_dataloader)
        
        # Log every epoch
        epoch_log = f"Epoch {epoch+1}/{num_epochs}: Average Loss: {avg_loss:.4f} [loss={last_loss:.4f}]"
        save_training_log(epoch_log, log_file)
        print(epoch_log)
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            checkpoint_log = f"Checkpoint saved: {checkpoint_path}"
            save_training_log(checkpoint_log, log_file)
    
    # Calculate training speed
    training_time = time.time() - start_time
    iterations_per_second = len(train_dataloader) * num_epochs / training_time
    
    # Final log entry
    final_log = f"""
Training Completed:
End Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Final Loss: {avg_loss:.4f}

Model Architecture:
- Multi-Query Attention (MLHA)
- Loss-less Load Balancing MoE
- DeepSeek-style architecture
- Pre-LayerNorm configuration
- Capacity Factor: 1.25

Training Summary:
- Total Epochs: {num_epochs}
- Final Average Loss: {avg_loss:.4f}
- Training Speed: ~{iterations_per_second:.2f} iterations/second
"""
    save_training_log(final_log, log_file)
    
    print(f"\nTraining log saved to: {log_file}") 