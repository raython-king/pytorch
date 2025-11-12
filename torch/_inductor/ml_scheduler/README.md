# ML-Based Scheduler for PyTorch Inductor

This directory contains the implementation of a machine learning-based scheduling system for PyTorch Inductor IR graphs.

## Overview

The ML scheduler uses Graph Neural Networks (GNNs), Transformers, and Reinforcement Learning to optimize:
- **Fusion decisions**: Which operations should be fused together
- **Scheduling order**: Optimal execution order for operations
- **Memory planning**: Efficient memory allocation and reuse

## Architecture

```
┌─────────────────────────────────────────┐
│    PyTorch Inductor Scheduler           │
│    (torch/_inductor/scheduler.py)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    ML Scheduler Orchestrator            │
│    • Feature Extraction                 │
│    • Model Inference                    │
│    • Confidence Scoring                 │
│    • Fallback Logic                     │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
┌─────────┐           ┌──────────┐
│   GNN   │           │ Transf.  │
│  Model  │           │  Model   │
└─────────┘           └──────────┘
```

## Directory Structure

```
ml_scheduler/
├── __init__.py              # Package initialization
├── README.md                # This file
├── DESIGN.md                # Comprehensive design document
├── orchestrator.py          # Main ML scheduler orchestrator
├── config.py                # Configuration management
│
├── models/                  # ML model implementations
│   ├── gnn_model.py         # Graph Neural Networks
│   ├── transformer_model.py # Transformer models
│   ├── rl_agent.py          # RL agent
│   ├── hybrid_model.py      # Hybrid architectures
│   └── ensemble.py          # Model ensemble
│
├── features/                # Feature extraction
│   ├── node_features.py     # Node-level features
│   ├── edge_features.py     # Edge-level features
│   ├── graph_features.py    # Graph-level features
│   └── feature_cache.py     # Caching system
│
├── training/                # Training infrastructure
│   ├── data_collector.py    # Collect training data
│   ├── dataset.py           # PyTorch Geometric dataset
│   ├── trainer.py           # Training orchestrator
│   ├── supervised.py        # Supervised learning
│   ├── rl_trainer.py        # RL training
│   └── curriculum.py        # Curriculum learning
│
├── inference/               # Model inference
│   ├── predictor.py         # Model inference
│   ├── confidence.py        # Confidence scoring
│   └── fallback.py          # Heuristic fallback
│
└── utils/                   # Utilities
    ├── graph_utils.py       # Graph manipulation
    ├── metrics.py           # Performance metrics
    └── visualization.py     # Visualization tools
```

## Usage

### Basic Usage

```python
from torch._inductor.ml_scheduler import MLSchedulerOrchestrator

# Initialize orchestrator
orchestrator = MLSchedulerOrchestrator()

# Predict fusion plan for nodes
fusion_plan = orchestrator.predict_fusion_plan(nodes, device)

# Check confidence
if fusion_plan.is_confident():
    # Apply ML-based fusion
    fused_nodes = apply_fusion_plan(nodes, fusion_plan)
else:
    # Fallback to heuristics
    fused_nodes = heuristic_fusion(nodes)
```

### Configuration

```python
from torch._inductor.ml_scheduler import MLSchedulerConfig

config = MLSchedulerConfig(
    model_path="/path/to/model.pt",
    confidence_threshold=0.75,
    cache_predictions=True,
)

orchestrator = MLSchedulerOrchestrator(config)
```

### Integration with Scheduler

Modify `torch/_inductor/scheduler.py`:

```python
class Scheduler:
    def fuse_nodes(self, nodes):
        if config.ml_scheduler_enabled:
            return self._ml_fuse_nodes(nodes)
        else:
            return self._heuristic_fuse_nodes(nodes)
    
    def _ml_fuse_nodes(self, nodes):
        from torch._inductor.ml_scheduler import MLSchedulerOrchestrator
        
        orchestrator = MLSchedulerOrchestrator()
        fusion_plan = orchestrator.predict_fusion_plan(nodes, self.current_device)
        
        if fusion_plan.is_confident():
            return self._apply_fusion_plan(nodes, fusion_plan)
        else:
            return self._heuristic_fuse_nodes(nodes)
```

## Training

### Data Collection

```python
from torch._inductor.ml_scheduler.training import DataCollector

collector = DataCollector()

# Collect traces from existing scheduler
dataset = collector.collect_from_benchmark(
    models=benchmark_models,
    num_samples=10000
)

# Save dataset
collector.save_dataset(dataset, "training_data.pt")
```

### Model Training

```python
from torch._inductor.ml_scheduler.training import SupervisedTrainer
from torch._inductor.ml_scheduler.models import FusionGNN

# Load data
dataset = torch.load("training_data.pt")

# Initialize model
model = FusionGNN(
    node_feat_dim=64,
    edge_feat_dim=32,
    hidden_dim=128,
)

# Train
trainer = SupervisedTrainer(model)
trainer.train(dataset, epochs=100)

# Save model
torch.save(model.state_dict(), "fusion_model.pt")
```

### RL Training

```python
from torch._inductor.ml_scheduler.training import RLTrainer
from torch._inductor.ml_scheduler.models import SchedulerRLAgent

agent = SchedulerRLAgent()
trainer = RLTrainer(agent)

# Train with PPO
trainer.train(num_iterations=1000)
```

## Evaluation

```python
from torch._inductor.ml_scheduler.utils import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# Evaluate on test set
results = evaluator.evaluate(
    test_set=test_graphs,
    baseline_scheduler=heuristic_scheduler,
    ml_scheduler=ml_scheduler,
)

print(f"Mean Speedup: {results['mean_speedup']:.2f}x")
print(f"Compilation Overhead: {results['compilation_overhead']:.2f}x")
print(f"Memory Efficiency: {results['memory_efficiency']:.2f}")
```

## Configuration Options

Enable ML scheduler in PyTorch Inductor config:

```python
import torch._inductor.config as config

# Enable ML scheduler
config.ml_scheduler_enabled = True
config.ml_scheduler_mode = "hybrid"  # "off", "shadow", "hybrid", "full"
config.ml_scheduler_model_path = "/path/to/model.pt"
config.ml_scheduler_confidence_threshold = 0.75
```

## Development Status

- [x] Design document
- [x] Core architecture
- [x] Feature extraction
- [ ] GNN model implementation (partial)
- [ ] Transformer model
- [ ] RL agent
- [ ] Training infrastructure
- [ ] Integration with scheduler.py
- [ ] Evaluation framework
- [ ] Production deployment

## Key Features

1. **Modular Design**: Each component can be developed and tested independently
2. **Safety First**: Extensive fallback mechanisms ensure correctness
3. **Incremental Deployment**: Can be deployed in shadow mode, then gradually rolled out
4. **Flexible Models**: Support for GNN, Transformer, RL, and hybrid architectures
5. **Online Learning**: Continuously improve from production workloads

## Performance Goals

- **Runtime Speedup**: 1.1-1.3x over heuristics
- **Compilation Overhead**: < 10% increase
- **Memory Efficiency**: No regression
- **Reliability**: > 99.9% correctness rate

## Contributing

See [DESIGN.md](DESIGN.md) for detailed architecture and design decisions.

## References

- PyTorch Inductor: https://pytorch.org/docs/stable/torch.compiler.html
- TorchDynamo: https://pytorch.org/docs/stable/dynamo/index.html
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
