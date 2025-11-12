# ML Scheduler Training Guide

This guide provides step-by-step instructions for training, validating, and deploying ML-based scheduler models for PyTorch Inductor.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Collection](#data-collection)
4. [Model Training](#model-training)
5. [Validation](#validation)
6. [Deployment](#deployment)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

## Overview

The ML scheduler learns to make optimal scheduling decisions (fusion, ordering, memory planning) by training on data collected from actual compilations. The training pipeline supports three modes:

- **Supervised Learning**: Learn from heuristic scheduler decisions
- **Imitation Learning**: Mimic expert traces
- **Reinforcement Learning**: Optimize for measured performance

## Prerequisites

### Software Requirements

```bash
# Core dependencies
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0

# Training dependencies
pip install tensorboard
pip install tqdm
pip install pandas

# Optional: for visualization
pip install matplotlib seaborn
```

### Hardware Requirements

- **Minimum**: CPU with 16GB RAM
- **Recommended**: GPU with 16GB+ VRAM for training
- **Optimal**: Multi-GPU setup for large-scale training

### Environment Setup

```bash
# Set environment variables
export PYTORCH_ML_SCHEDULER_DATA_DIR=/path/to/data
export PYTORCH_ML_SCHEDULER_MODEL_DIR=/path/to/models

# Enable debug logging
export TORCH_LOGS="+inductor,ml_scheduler"
```

## Data Collection

### Step 1: Collect Training Data

Use the existing heuristic scheduler to collect training data from real compilations:

```python
from torch._inductor.ml_scheduler.examples.train_and_deploy import collect_training_data
from pathlib import Path

# Collect 10,000 graph samples
output_dir = Path("./training_data")
collect_training_data(output_dir, num_samples=10000)
```

Or use the CLI:

```bash
python -m torch._inductor.ml_scheduler.examples.train_and_deploy \
    --mode collect \
    --output ./training_data \
    --num-samples 10000
```

### Step 2: Collect from Real Workloads

For better performance, collect data from your actual workloads:

```python
import torch
from torch._inductor.ml_scheduler.orchestrator import MLSchedulerOrchestrator

# Enable data collection mode
config = MLSchedulerConfig(collect_feedback=True)
orchestrator = MLSchedulerOrchestrator(config=config)

# Run your normal compilation workloads
model = torch.compile(your_model)
output = model(input_data)

# Data will be collected in feedback buffer
# Save to disk for training
```

### Data Format

Training data consists of IR graphs with labels:

```python
{
    'x': torch.Tensor,              # Node features [num_nodes, 64]
    'edge_index': torch.Tensor,     # Edge connectivity [2, num_edges]
    'edge_attr': torch.Tensor,      # Edge features [num_edges, 32]
    'num_nodes': int,               # Number of nodes
    'y_fusion': torch.Tensor,       # Fusion labels [num_nodes, num_nodes]
    'y_schedule': torch.Tensor,     # Schedule order [num_nodes]
    'y_performance': float,         # Performance metric (optional)
}
```

## Model Training

### Basic Training

```python
from torch._inductor.ml_scheduler.training.trainer import MLSchedulerTrainer, TrainingConfig
from torch._inductor.ml_scheduler.training.dataset import create_dataset_split

# Create dataset split
train_dataset, val_dataset, test_dataset = create_dataset_split(
    data_path='./training_data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)

# Configure training
config = TrainingConfig(
    mode='supervised',
    model_type='fusion_gnn',
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    checkpoint_dir='./checkpoints',
    log_dir='./logs',
)

# Create trainer
trainer = MLSchedulerTrainer(
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
)

# Train
trainer.train()
```

Or use the CLI:

```bash
python -m torch._inductor.ml_scheduler.examples.train_and_deploy \
    --mode train \
    --data ./training_data \
    --output ./checkpoints \
    --num-epochs 100 \
    --batch-size 32
```

### Training Modes

#### 1. Supervised Learning (Recommended for First Iteration)

Learn from heuristic scheduler decisions:

```python
config = TrainingConfig(
    mode='supervised',
    fusion_loss_weight=1.0,
    schedule_loss_weight=0.5,
    performance_loss_weight=2.0,
)
```

**Advantages:**
- Easy to get started
- Stable training
- Good baseline performance

**Disadvantages:**
- Limited by heuristic quality
- May not discover better strategies

#### 2. Imitation Learning

Learn from expert traces (manually optimized schedules):

```python
config = TrainingConfig(
    mode='imitation',
    # Load expert demonstrations
)
```

**Use when:**
- You have manually optimized schedules
- Heuristics are suboptimal
- Need to learn complex patterns

#### 3. Reinforcement Learning

Optimize for actual performance metrics:

```python
config = TrainingConfig(
    mode='rl',
    rl_discount_factor=0.99,
    rl_entropy_coef=0.01,
    rl_value_loss_coef=0.5,
)
```

**Use when:**
- Have performance measurement environment
- Want to optimize end-to-end
- Can afford longer training time

### Multi-GPU Training

```python
# Enable distributed training
config = TrainingConfig(
    use_distributed=True,
    world_size=4,  # Number of GPUs
)

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open browser to http://localhost:6006
```

Monitor these metrics:
- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease without overfitting
- **Learning rate**: Should follow schedule
- **Gradient norms**: Should be stable (not exploding/vanishing)

## Validation

### Step 1: Shadow Mode Validation

Test the model without affecting actual compilation:

```python
from torch._inductor.ml_scheduler.integration.scheduler_hook import (
    enable_ml_scheduler,
    MLSchedulerMode,
)

enable_ml_scheduler(
    mode=MLSchedulerMode.SHADOW,
    config=MLSchedulerConfig(
        model_path='./checkpoints/best_model.pt',
        confidence_threshold=0.75,
    )
)

# Run your normal workloads
# ML predictions will be logged but not applied
```

Or use the CLI:

```bash
python -m torch._inductor.ml_scheduler.examples.train_and_deploy \
    --mode validate \
    --model ./checkpoints/best_model.pt
```

### Step 2: Analyze Shadow Logs

```python
from torch._inductor.ml_scheduler.integration.scheduler_hook import MLSchedulerWrapper

# Get shadow logs
wrapper = MLSchedulerWrapper(scheduler, mode=MLSchedulerMode.SHADOW)
logs = wrapper.get_shadow_log()

# Analyze predictions
for log in logs:
    print(f"Nodes: {log['num_nodes']}")
    print(f"Fusions: {len(log['fusion_plan'].fusions)}")
    print(f"Confidence: {sum(log['fusion_plan'].confidence_scores)/len(log['fusion_plan'].confidence_scores)}")
```

### Step 3: A/B Testing

Compare ML vs heuristic on test workloads:

```python
from torch._inductor.ml_scheduler.integration.scheduler_hook import compare_ml_vs_heuristic

results = compare_ml_vs_heuristic(
    nodes=test_nodes,
    scheduler=scheduler,
    config=config,
)

print(f"ML nodes: {results['num_nodes_ml']}")
print(f"Heuristic nodes: {results['num_nodes_heuristic']}")
print(f"Time ratio: {results['time_ratio']:.2f}x")
```

## Deployment

### Hybrid Mode (Recommended)

Use ML when confident, fallback to heuristics otherwise:

```python
enable_ml_scheduler(
    mode=MLSchedulerMode.HYBRID,
    config=MLSchedulerConfig(
        model_path='./checkpoints/best_model.pt',
        confidence_threshold=0.75,
        fallback_on_error=True,
        validate_fusion_plan=True,
        max_inference_time_ms=50.0,
    )
)
```

Or use the CLI:

```bash
python -m torch._inductor.ml_scheduler.examples.train_and_deploy \
    --mode deploy \
    --model ./checkpoints/best_model.pt \
    --deploy-mode hybrid
```

### Full Mode (Advanced)

Use ML exclusively (only after thorough validation):

```python
enable_ml_scheduler(
    mode=MLSchedulerMode.FULL,
    config=MLSchedulerConfig(
        model_path='./checkpoints/best_model.pt',
        confidence_threshold=0.6,
    )
)
```

### Production Deployment Checklist

- [ ] Validated in shadow mode for 1000+ compilations
- [ ] A/B testing shows improvement over heuristics
- [ ] Inference time overhead < 10% of compilation time
- [ ] Fallback mechanisms tested and working
- [ ] Memory usage is acceptable
- [ ] Model checkpoints are backed up
- [ ] Monitoring and alerting configured
- [ ] Rollback plan prepared

## Hyperparameter Tuning

### Model Architecture

```python
config = TrainingConfig(
    # Model size
    hidden_dim=128,        # Try: 64, 128, 256
    num_layers=4,          # Try: 2, 4, 6
    num_heads=4,           # Try: 2, 4, 8
    dropout=0.1,           # Try: 0.0, 0.1, 0.2
)
```

**Guidelines:**
- Larger models: Better performance but slower inference
- More layers: Better for complex graphs
- More heads: Better attention but more memory
- Higher dropout: Reduces overfitting

### Training Hyperparameters

```python
config = TrainingConfig(
    # Optimization
    learning_rate=1e-3,    # Try: 1e-4, 1e-3, 1e-2
    weight_decay=1e-4,     # Try: 0, 1e-5, 1e-4
    batch_size=32,         # Try: 16, 32, 64

    # Learning rate schedule
    lr_scheduler='cosine', # Try: 'cosine', 'step', 'plateau'
    warmup_epochs=5,       # Try: 0, 5, 10

    # Regularization
    grad_clip_norm=1.0,    # Try: 0.5, 1.0, 5.0
)
```

### Loss Weights

```python
config = TrainingConfig(
    fusion_loss_weight=1.0,      # Adjust based on importance
    schedule_loss_weight=0.5,
    performance_loss_weight=2.0,
)
```

### Grid Search Example

```python
from itertools import product

# Define hyperparameter grid
hidden_dims = [64, 128, 256]
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]

best_val_loss = float('inf')
best_config = None

for hidden_dim, lr, batch_size in product(hidden_dims, learning_rates, batch_sizes):
    config = TrainingConfig(
        hidden_dim=hidden_dim,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=50,
    )

    trainer = MLSchedulerTrainer(config, train_dataset, val_dataset)
    trainer.train()

    if trainer.best_val_loss < best_val_loss:
        best_val_loss = trainer.best_val_loss
        best_config = config

print(f"Best config: {best_config}")
```

## Troubleshooting

### Common Issues

#### 1. Training Loss Not Decreasing

**Symptoms:** Loss stays constant or decreases very slowly

**Solutions:**
- Increase learning rate (try 10x higher)
- Reduce model complexity (fewer layers/smaller hidden dim)
- Check for bugs in loss computation
- Verify data labels are correct
- Try different optimizer (Adam → SGD or vice versa)

#### 2. Overfitting

**Symptoms:** Training loss decreases but validation loss increases

**Solutions:**
- Increase dropout (0.1 → 0.3)
- Add more training data
- Reduce model size
- Use stronger regularization (higher weight_decay)
- Enable data augmentation
- Early stopping (reduce patience)

#### 3. Slow Inference

**Symptoms:** ML inference takes too long, causing compilation slowdown

**Solutions:**
- Reduce model size (smaller hidden_dim, fewer layers)
- Enable model quantization
- Use batch inference
- Increase confidence threshold (skip more graphs)
- Profile and optimize bottlenecks

#### 4. Low Confidence Scores

**Symptoms:** Model always falls back to heuristics

**Solutions:**
- Train longer (more epochs)
- Collect more training data
- Lower confidence threshold
- Check for distribution mismatch between train/test
- Improve model architecture

#### 5. Memory Errors

**Symptoms:** OOM during training or inference

**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training (use_amp=True)
- Reduce model size
- Use gradient checkpointing
- Filter out very large graphs (max_nodes_for_ml)

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile inference time
from torch._inductor.ml_scheduler.examples.train_and_deploy import benchmark_model
benchmark_model('./checkpoints/best_model.pt', num_iterations=100)

# Visualize model predictions
def visualize_fusion_matrix(fusion_matrix):
    import matplotlib.pyplot as plt
    plt.imshow(fusion_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Fusion Matrix')
    plt.show()

# Check model outputs
with torch.no_grad():
    output = model(x, edge_index, edge_attr)
    print(f"Fusion matrix range: [{output['fusion_matrix'].min():.3f}, {output['fusion_matrix'].max():.3f}]")
    print(f"Mean confidence: {output['fusion_matrix'].mean():.3f}")
```

## Performance Optimization

### Inference Optimization

1. **Model Quantization**
   ```python
   # Quantize model for faster inference
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Batch Inference**
   ```python
   # Process multiple graphs at once
   predictor.predict_batch(batch_data)
   ```

3. **Result Caching**
   ```python
   config = MLSchedulerConfig(
       cache_predictions=True,
       cache_size=1000,
   )
   ```

4. **Early Exit**
   ```python
   config = MLSchedulerConfig(
       max_inference_time_ms=50.0,  # Timeout after 50ms
   )
   ```

### Training Optimization

1. **Mixed Precision Training**
   ```python
   config = TrainingConfig(use_amp=True)
   ```

2. **Gradient Accumulation**
   ```python
   # Effective batch size = batch_size * accumulation_steps
   for i, batch in enumerate(dataloader):
       loss = compute_loss(batch)
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Data Loading Optimization**
   ```python
   dataloader = DataLoader(
       dataset,
       num_workers=4,      # Parallel data loading
       pin_memory=True,    # Faster GPU transfer
       prefetch_factor=2,  # Prefetch batches
   )
   ```

### Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Inference time | < 10ms | < 50ms | > 100ms |
| Compilation overhead | < 5% | < 10% | > 20% |
| Model accuracy | > 90% | > 80% | < 70% |
| Cache hit rate | > 50% | > 30% | < 10% |
| Memory usage | < 500MB | < 1GB | > 2GB |

## Best Practices

1. **Start Simple**
   - Begin with supervised learning on heuristic data
   - Use default hyperparameters
   - Validate in shadow mode extensively

2. **Iterate Gradually**
   - Collect more data as you go
   - Tune hyperparameters systematically
   - Deploy in hybrid mode first

3. **Monitor Continuously**
   - Track inference time overhead
   - Monitor fallback rate
   - Log prediction confidence
   - Alert on anomalies

4. **Version Control**
   - Tag model versions
   - Keep training data versioned
   - Document hyperparameter changes
   - Maintain model changelog

5. **Safety First**
   - Always have fallback to heuristics
   - Set aggressive timeout limits
   - Validate fusion plans
   - Test thoroughly before production

## Further Reading

- [DESIGN.md](DESIGN.md) - System architecture and design
- [README.md](README.md) - Overview and quick start
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration details
- [examples/train_and_deploy.py](examples/train_and_deploy.py) - Complete example

## Support

For issues and questions:
- GitHub Issues: [pytorch/pytorch](https://github.com/pytorch/pytorch/issues)
- PyTorch Forums: [discuss.pytorch.org](https://discuss.pytorch.org)
- Documentation: [pytorch.org/docs](https://pytorch.org/docs)
