import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MLHAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # DeepSeek style MLHA
        self.num_key_value_heads = 1  # Single KV head for multi-query
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, 1, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, 1, self.head_dim)
        
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            # Expand attention mask to match attention scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_length, seq_length)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, v)
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, self.hidden_size)
        
        return self.o_proj(output)

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.capacity_factor = 1.25
        
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.ffn_dim),
                nn.GELU(),
                nn.Linear(self.ffn_dim, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        tokens_per_expert = (batch_size * seq_len) // self.num_experts
        capacity = int(self.capacity_factor * tokens_per_expert)
        
        # Reshape input for routing
        hidden_states_reshaped = hidden_states.view(-1, hidden_size)
        
        # Get routing weights
        gate_logits = self.gate(hidden_states_reshaped)
        routing_weights = F.softmax(gate_logits, dim=-1)
        
        # Initialize output tensor
        final_output = torch.zeros_like(hidden_states_reshaped)
        total_tokens = routing_weights.size(0)
        
        # Process each expert
        load_balancing_losses = []
        for i in range(self.num_experts):
            # Get scores for current expert
            expert_scores = routing_weights[:, i]
            
            # Sort scores
            scores_sorted, indices = torch.sort(expert_scores, descending=True)
            
            # Select top-k tokens within capacity
            top_k_indices = indices[:capacity]
            top_k_scores = scores_sorted[:capacity]
            
            # Compute load balancing loss for this expert
            expert_load = top_k_scores.sum() / total_tokens
            load_balancing_losses.append(expert_load)
            
            if top_k_indices.size(0) > 0:
                # Process selected tokens through expert
                expert_input = hidden_states_reshaped[top_k_indices]
                expert_output = self.experts[i](expert_input)
                
                # Add weighted output to final result
                final_output[top_k_indices] += expert_output * top_k_scores.unsqueeze(-1)
        
        # Compute final load balancing loss
        load_balancing_loss = torch.var(torch.stack(load_balancing_losses)) * self.num_experts
        
        # Reshape output back to original dimensions
        output = final_output.view(batch_size, seq_len, hidden_size)
        
        return output, load_balancing_loss

class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln_1': nn.LayerNorm(config.hidden_size),
                'mlha': MLHAAttention(config),
                'ln_2': nn.LayerNorm(config.hidden_size),
                'moe': MoELayer(config)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        # Add language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embedding(input_ids)
        total_load_balancing_loss = 0
        
        for layer in self.layers:
            # MLHA
            residual = hidden_states
            hidden_states = layer['ln_1'](hidden_states)
            hidden_states = layer['mlha'](hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            
            # MoE
            residual = hidden_states
            hidden_states = layer['ln_2'](hidden_states)
            hidden_states, lb_loss = layer['moe'](hidden_states)
            hidden_states = residual + hidden_states
            total_load_balancing_loss += lb_loss
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, total_load_balancing_loss

    def generate(self, input_ids, attention_mask=None, max_length=100, temperature=0.7, top_p=0.9):
        self.eval()
        
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits directly from forward pass
                logits, _ = self.forward(input_ids, attention_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Filter pad, bos, eos tokens
                next_token_logits[:, [0, 1, 2]] = -float('inf')
                
                # Apply nucleus sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we predict EOS
                if (next_token == 2).any():
                    break
                    
                # Concatenate next token
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_ones((batch_size, 1))
                    ], dim=1)
                
                current_length += 1
                
                # Stop if too long
                if current_length >= max_length:
                    break
        
        return input_ids 