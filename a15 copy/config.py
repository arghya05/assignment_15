class DeepSeekConfig:
    def __init__(self):
        # Model architecture (reduced for 8GB RAM)
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_hidden_layers = 6
        self.intermediate_size = 1024
        self.num_experts = 2
        
        # DeepSeek specific settings
        self.max_position_embeddings = 512
        self.vocab_size = 50257  # GPT-2 vocabulary size
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02
        self.use_cache = True
        
        # Training settings
        self.batch_size = 2
        self.sequence_length = 128
        self.gradient_checkpointing = False  # Enable if needed
        
        # MoE settings
        self.moe_capacity_factor = 1.25
        self.moe_loss_weight = 0.01 