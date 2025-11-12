# ML-Based Scheduling System: Executive Summary

## Overview

This document provides a high-level summary of the comprehensive ML-based scheduling system designed for PyTorch Inductor IR graphs.

## What Problem Does It Solve?

Current PyTorch Inductor uses hand-crafted heuristics for:
- Deciding which operations to fuse together
- Determining optimal execution order
- Planning memory allocation and reuse

These heuristics are:
- Limited by human intuition
- Hard to tune across diverse workloads
- May miss non-obvious optimization opportunities

**Solution**: Use machine learning to learn optimal scheduling decisions from data.

## Key Components

### 1. Model Architecture (Choose One or Ensemble)

| Model Type | Best For | Pros | Cons |
|-----------|----------|------|------|
| **GNN** | Local fusion decisions | Captures graph structure, efficient | Limited global context |
| **Transformer** | Sequential scheduling | Long-range dependencies, flexible | Higher compute cost |
| **RL Agent** | Direct optimization | Optimizes for performance | Requires environment |
| **Hybrid** | Best of both | Robust, high accuracy | More complex |

### 2. Feature Engineering

Extract rich features from IR graphs:

**Node Features (64 dims)**:
- Operation type, FLOPs, memory access
- Data dependencies, device properties
- Scheduling constraints

**Edge Features (32 dims)**:
- Dependency type, data size
- Transfer cost, fusion compatibility

**Graph Features (32 dims)**:
- Graph statistics, resource constraints
- Workload characteristics

### 3. Training Strategy

**Phase 1: Supervised Learning**
- Learn from existing heuristic decisions
- Fast to train, stable baseline
- Bootstrap the model

**Phase 2: Reinforcement Learning**
- Optimize directly for performance
- Discover novel optimizations
- Requires more compute

**Phase 3: Online Learning**
- Continuously improve from production
- Adapt to new workloads
- Gradual improvement

### 4. Integration Points

Minimal changes to existing code:

```python
# torch/_inductor/scheduler.py

class Scheduler:
    def fuse_nodes(self, nodes):
        if config.ml_scheduler_enabled:
            return self._ml_fuse_nodes(nodes)  # NEW
        else:
            return self._heuristic_fuse_nodes(nodes)  # EXISTING
```

### 5. Safety and Fallback

Multiple layers of safety:
1. **Confidence scoring**: Only use high-confidence predictions
2. **Validation**: Check correctness before applying
3. **Heuristic fallback**: Revert to heuristics on failure
4. **Shadow mode**: Test without affecting results

## File Structure

```
torch/_inductor/ml_scheduler/
├── DESIGN.md                 # Comprehensive design (50+ pages)
├── README.md                 # User guide
├── INTEGRATION_GUIDE.md      # Integration instructions
├── SUMMARY.md                # This file
│
├── orchestrator.py           # Main entry point
├── config.py                 # Configuration
│
├── models/                   # ML models
│   ├── gnn_model.py
│   ├── transformer_model.py
│   ├── rl_agent.py
│   └── ensemble.py
│
├── features/                 # Feature extraction
│   ├── node_features.py
│   ├── edge_features.py
│   └── graph_features.py
│
├── training/                 # Training infrastructure
├── inference/                # Model inference
└── utils/                    # Utilities
```

## Quick Start

### 1. Enable ML Scheduler

```bash
export TORCHINDUCTOR_ML_SCHEDULER=1
export TORCHINDUCTOR_ML_SCHEDULER_MODE=shadow  # Safe testing
```

### 2. Train Model (Optional)

```python
from torch._inductor.ml_scheduler.training import SupervisedTrainer
from torch._inductor.ml_scheduler.models import FusionGNN

# Collect data
dataset = collect_training_data(num_samples=10000)

# Train model
model = FusionGNN()
trainer = SupervisedTrainer(model)
trainer.train(dataset, epochs=100)

# Save model
torch.save(model.state_dict(), "fusion_model.pt")
```

### 3. Use in Production

```bash
export TORCHINDUCTOR_ML_SCHEDULER=1
export TORCHINDUCTOR_ML_SCHEDULER_MODE=hybrid
export TORCHINDUCTOR_ML_SCHEDULER_MODEL_PATH=/path/to/fusion_model.pt
```

## Performance Goals

| Metric | Target | Status |
|--------|--------|--------|
| Runtime Speedup | 1.1-1.3x | Design phase |
| Compilation Overhead | < 10% | Design phase |
| Memory Efficiency | No regression | Design phase |
| Reliability | > 99.9% | Design phase |

## Deployment Plan

```
Phase 1: Data Collection (Month 1-2)
  └─> Collect 10K+ traces from existing scheduler

Phase 2: Model Development (Month 3-4)
  └─> Train and evaluate models offline

Phase 3: Shadow Mode (Month 5)
  └─> Run in parallel, collect metrics

Phase 4: Canary Deployment (Month 6)
  └─> 1% -> 10% -> 50% traffic

Phase 5: Full Deployment (Month 7-8)
  └─> 100% with fallback

Phase 6: Optimization (Month 9+)
  └─> Online learning, model compression
```

## Key Design Decisions

### Why Hybrid Approach?

- **GNN** excels at local structure (which ops to fuse)
- **Transformer** excels at global patterns (execution order)
- **Combining both** gives best results

### Why Incremental Deployment?

- **Safety**: Extensive testing before production
- **Validation**: Gather data to improve models
- **Confidence**: Build trust gradually

### Why Support Fallback?

- **Reliability**: Never break existing functionality
- **Flexibility**: Use ML only when confident
- **Safety net**: Handle edge cases gracefully

## Evaluation Metrics

### Primary Metrics
- **Runtime performance**: Speedup vs heuristics
- **Compilation time**: Overhead of ML inference
- **Memory usage**: Peak memory consumption

### Secondary Metrics
- **Generalization**: Performance on unseen graphs
- **Reliability**: Correctness rate
- **Robustness**: Handling of edge cases

## Technical Challenges

### 1. Feature Engineering
**Challenge**: Extract meaningful features from IR graphs
**Solution**: Comprehensive feature extractors with domain knowledge

### 2. Training Data
**Challenge**: Limited labeled data
**Solution**: Bootstrap from heuristics, use RL for improvement

### 3. Compilation Speed
**Challenge**: ML inference adds overhead
**Solution**: Model compression, caching, fast inference

### 4. Generalization
**Challenge**: Diverse graph types and sizes
**Solution**: Curriculum learning, ensemble models

### 5. Safety
**Challenge**: Cannot break correctness
**Solution**: Multiple validation layers, fallback mechanisms

## Research Opportunities

1. **Neural Architecture Search**: Automatically design model architectures
2. **Multi-Task Learning**: Joint optimization of fusion and scheduling
3. **Transfer Learning**: Adapt to new hardware/workloads
4. **Interpretability**: Understand why certain decisions are made
5. **Active Learning**: Intelligently select training samples

## Comparison with Alternatives

| Approach | Pros | Cons | Our Solution |
|----------|------|------|--------------|
| Hand-crafted heuristics | Fast, predictable | Limited, hard to tune | Augment, not replace |
| Exhaustive search | Optimal | Slow, impractical | Use ML to guide search |
| Genetic algorithms | Exploratory | Slow, random | Use ML for directed exploration |
| ML end-to-end | Automated | Complex, risky | Gradual integration with fallback |

## Success Criteria

### Must Have
- [ ] No correctness regressions
- [ ] Compilation time overhead < 20%
- [ ] Works on diverse workloads

### Should Have
- [ ] Runtime speedup > 1.1x on average
- [ ] Memory efficiency maintained
- [ ] Generalization to unseen graphs

### Nice to Have
- [ ] Runtime speedup > 1.3x on some workloads
- [ ] Online learning working
- [ ] Explainable decisions

## Resources Required

### Compute
- **Training**: ~100 GPU-hours for initial training
- **Data Collection**: ~1000 CPU-hours for trace collection
- **Evaluation**: ~50 GPU-hours for benchmarking

### Engineering
- **Core Development**: 2-3 engineers × 3-4 months
- **Integration**: 1 engineer × 1 month
- **Testing & Deployment**: 1-2 engineers × 2 months

### Infrastructure
- Model storage and versioning
- Monitoring and metrics collection
- A/B testing framework

## Next Steps

1. **Review Design** (Week 1)
   - Get feedback from team
   - Refine architecture

2. **Implement Feature Extractors** (Week 2-3)
   - Node, edge, graph features
   - Unit tests

3. **Collect Training Data** (Week 4-5)
   - Instrument existing scheduler
   - Collect 10K+ traces

4. **Train Initial Models** (Week 6-8)
   - GNN baseline
   - Transformer baseline
   - Evaluate offline

5. **Shadow Mode Testing** (Week 9-10)
   - Run in parallel
   - Collect metrics
   - Compare with heuristics

6. **Gradual Rollout** (Week 11-16)
   - 1% -> 10% -> 50% -> 100%
   - Monitor for issues
   - Iterate on feedback

## Conclusion

This ML-based scheduling system provides:
- **Modular design** for incremental development
- **Safety mechanisms** to ensure correctness
- **Flexible architecture** supporting multiple model types
- **Clear integration path** with existing code
- **Comprehensive evaluation** framework

The system is designed to augment, not replace, existing heuristics, allowing gradual adoption with minimal risk.

---

## References

- [DESIGN.md](DESIGN.md) - Comprehensive design document
- [README.md](README.md) - User guide and API documentation
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration instructions

## Contact

For questions or feedback, please contact the PyTorch Inductor team.
