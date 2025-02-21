import torch
from transformers import AutoTokenizer
from model import DeepSeekBlock
from config import DeepSeekConfig

def generate_outputs(checkpoint_path, prompts, device='mps'):
    # Initialize model and load checkpoint
    config = DeepSeekConfig()
    model = DeepSeekBlock(config)
    
    # Load checkpoint with strict=False to handle missing keys
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Info: Missing keys when loading checkpoint: {missing_keys}")
    
    # Initialize lm_head weights if missing
    if hasattr(model, 'lm_head'):
        if 'lm_head.weight' not in checkpoint['model_state_dict']:
            print("Initializing language modeling head...")
            # Initialize from embedding weights for better starting point
            model.lm_head.weight.data = model.embedding.weight.data.clone()
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    generated_outputs = []
    
    for prompt in prompts:
        print(f"\nGenerating for prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=config.sequence_length,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate with lower temperature for more focused outputs
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=200,
            temperature=0.6,
            top_p=0.9
        )
        
        # Decode output properly
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_outputs.append(generated_text)
        
        print(f"Generated: {generated_text}\n")
        print("-" * 50)
    
    return generated_outputs

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch_1000.pt"  # Use your latest checkpoint
    
    prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humans will",
        "The most important scientific discovery of the 21st century was",
        "The relationship between technology and society has",
        "Climate change will affect the world by"
    ]
    
    try:
        generated_outputs = generate_outputs(checkpoint_path, prompts)
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print("\nTrying to retrain the model...")
        
        # If generation fails, suggest retraining
        print("""
Please retrain the model with the updated architecture:
1. Delete old checkpoints
2. Run: python main.py
3. After training completes, run: python generate.py
""") 