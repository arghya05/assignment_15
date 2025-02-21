# DeepSeek Architecture Implementation with MLHA and MoE

This repository contains an implementation of the DeepSeek architecture with Multi-Query Head Attention (MLHA) and Mixture of Experts (MoE) with Loss-less load balancing, optimized for M2 Mac with 8GB RAM.

## Training Results

### Model Configuration
- Hidden Size: 256
- Number of Attention Heads: 8
- Number of Hidden Layers: 6
- Intermediate Size: 1024
- Number of Experts: 2
- Batch Size: 2
- Sequence Length: 128
- Gradient Accumulation Steps: 8

### Training Performance
- Training Device: M2 Mac (MPS)
- Average Speed: ~25 iterations/second
- Final Loss: -1.2374

### Training Progress
The model was trained for 1000 epochs, showing consistent loss reduction:

Epoch 989/1000: Average Loss: -1.2238 [loss=-0.1531]
Epoch 990/1000: Average Loss: -1.2250 [loss=-0.1532]
Epoch 991/1000: Average Loss: -1.2263 [loss=-0.1534]
Epoch 992/1000: Average Loss: -1.2275 [loss=-0.1535]
Epoch 993/1000: Average Loss: -1.2287 [loss=-0.1537]
Epoch 994/1000: Average Loss: -1.2300 [loss=-0.1538]
Epoch 995/1000: Average Loss: -1.2312 [loss=-0.1540]
Epoch 996/1000: Average Loss: -1.2324 [loss=-0.1541]
Epoch 997/1000: Average Loss: -1.2337 [loss=-0.1543]
Epoch 998/1000: Average Loss: -1.2349 [loss=-0.1544]
Epoch 999/1000: Average Loss: -1.2361 [loss=-0.1546]
Epoch 1000/1000: Average Loss: -1.2374 [loss=-0.1547]


## Generation Results

The model was tested with five different prompts. Here are the generation results:

1. **Prompt**: "The future of artificial intelligence is"
   - Generated text shows mixed coherence with some technical terms but needs improvement in fluency

2. **Prompt**: "In the year 2050, humans will"
   - Output contains fragmented phrases and needs better context handling

3. **Prompt**: "The most important scientific discovery of the 21st century was"
   - Generation shows some scientific vocabulary but lacks coherent narrative

4. **Prompt**: "The relationship between technology and society has"
   - Output includes relevant terms but needs better sentence structure

5. **Prompt**: "Climate change will affect the world by"
   - Generated text contains environmental terms but requires better semantic coherence

## Model Architecture Features

1. **Multi-Query Head Attention (MLHA)**
   - Single KV head for multi-query attention
   - Efficient memory usage through head sharing
   - Scaled dot-product attention with mask support

2. **Mixture of Experts (MoE)**
   - Loss-less load balancing implementation
   - Capacity factor: 1.25
   - Dynamic expert routing
   - Load balancing loss computation

3. **Architecture Optimizations**
   - Pre-LayerNorm configuration
   - Gradient accumulation (steps=8)
   - Memory-efficient implementation for M2 Mac
   - Proper token handling and generation

## Areas for Improvement

1. Text Generation Quality
   - Implement better temperature scheduling
   - Improve token filtering
   - Add beam search for better coherence

2. Model Architecture
   - Fine-tune the number of experts
   - Adjust capacity factor
   - Optimize attention mechanism

3. Training Process
   - Increase training data size
   - Implement learning rate scheduling
   - Add validation metrics

## Dependencies
- torch
- transformers
- tqdm
- datasets
- numpy

## Training Logs
Full training logs are available at: `logs/training_log_20250210_220424.txt`