# Integration Guide: ML Scheduler with PyTorch Inductor

This guide provides step-by-step instructions for integrating the ML scheduler with the existing PyTorch Inductor scheduler.

## Integration Points

### 1. Main Scheduler Integration (scheduler.py)

#### Location: `torch/_inductor/scheduler.py`

Add ML scheduler initialization in `Scheduler.__init__()`:

```python
class Scheduler:
    def __init__(self, nodes: list[ir.Operation]) -> None:
        # ... existing initialization ...
        
        # Initialize ML scheduler if enabled
        if config.ml_scheduler_enabled:
            from torch._inductor.ml_scheduler import MLSchedulerOrchestrator
            try:
                self.ml_orchestrator = MLSchedulerOrchestrator(
                    config=self._get_ml_config()
                )
                log.info("ML Scheduler initialized successfully")
            except Exception as e:
                log.warning(f"Failed to initialize ML Scheduler: {e}")
                self.ml_orchestrator = None
        else:
            self.ml_orchestrator = None
    
    def _get_ml_config(self):
        """Get ML scheduler configuration from inductor config."""
        from torch._inductor.ml_scheduler import MLSchedulerConfig
        
        return MLSchedulerConfig(
            model_path=config.ml_scheduler_model_path,
            confidence_threshold=config.ml_scheduler_confidence_threshold,
            cache_predictions=config.ml_scheduler_cache_predictions,
            collect_feedback=config.ml_scheduler_collect_feedback,
        )
```

#### Modify `fuse_nodes()` method:

```python
def fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Combine eligible nodes into FusedSchedulerNodes.
    
    Now supports ML-based fusion decisions.
    """
    # Check if ML scheduler should be used
    if self._should_use_ml_scheduler(nodes):
        return self._ml_fuse_nodes(nodes)
    else:
        return self._heuristic_fuse_nodes(nodes)

def _should_use_ml_scheduler(self, nodes: list[BaseSchedulerNode]) -> bool:
    """Decide whether to use ML scheduler."""
    if not config.ml_scheduler_enabled or self.ml_orchestrator is None:
        return False
    
    # Only use for graphs of reasonable size
    if len(nodes) < config.ml_scheduler_min_nodes:
        return False
    
    if len(nodes) > config.ml_scheduler_max_nodes:
        return False
    
    # Shadow mode: run ML but don't use results
    if config.ml_scheduler_mode == "shadow":
        self._ml_fuse_nodes_shadow(nodes)
        return False
    
    return config.ml_scheduler_mode in ["hybrid", "full"]

def _ml_fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """ML-based fusion with heuristic fallback."""
    try:
        # Get ML predictions
        with dynamo_timed("ml_scheduler.predict"):
            fusion_plan = self.ml_orchestrator.predict_fusion_plan(
                nodes=nodes,
                device=self.current_device
            )
        
        # Validate fusion plan
        is_valid, error = self._validate_fusion_plan(fusion_plan, nodes)
        if not is_valid:
            log.warning(f"ML fusion plan invalid: {error}, using heuristics")
            return self._heuristic_fuse_nodes(nodes)
        
        # Check confidence
        avg_confidence = (
            sum(fusion_plan.confidence_scores) / len(fusion_plan.confidence_scores)
            if fusion_plan.confidence_scores else 0.0
        )
        
        if avg_confidence < config.ml_scheduler_confidence_threshold:
            log.debug(
                f"ML confidence too low ({avg_confidence:.3f}), using heuristics"
            )
            return self._heuristic_fuse_nodes(nodes)
        
        # Apply ML fusion plan
        fused_nodes = self._apply_fusion_plan(nodes, fusion_plan)
        
        # Record for feedback
        if config.ml_scheduler_collect_feedback:
            self.ml_orchestrator.record_prediction(
                nodes=nodes,
                fusion_plan=fusion_plan,
                result_nodes=fused_nodes
            )
        
        fusion_log.info(
            f"ML scheduler: fused {len(nodes)} -> {len(fused_nodes)} nodes "
            f"(confidence: {avg_confidence:.3f})"
        )
        
        return fused_nodes
        
    except Exception as e:
        log.error(f"ML scheduler failed: {e}, falling back to heuristics")
        if config.ml_scheduler_fallback_on_error:
            return self._heuristic_fuse_nodes(nodes)
        else:
            raise

def _heuristic_fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Original heuristic-based fusion (unchanged)."""
    with dynamo_timed(
        "Scheduler.fused_nodes", log_pt2_compile_event=True, log_waitcounter=True
    ):
        for i in range(10):
            old_len = len(nodes)
            fusion_log.debug(
                "===== attempting fusion (%d/10): %d nodes =====",
                i + 1,
                old_len,
            )
            nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
            new_len = len(nodes)
            fusion_log.debug(
                "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
                i + 1,
                old_len,
                new_len,
            )
            if new_len == old_len or new_len == 1:
                fusion_log.debug(
                    "===== fusion complete (%d iterations) =====", i + 1
                )
                break
        
        if (
            config.loop_ordering_after_fusion
            or config.loop_index_inversion_in_fusion
        ):
            nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
        return nodes

def _apply_fusion_plan(
    self,
    nodes: list[BaseSchedulerNode],
    fusion_plan: FusionPlan
) -> list[BaseSchedulerNode]:
    """
    Apply fusion plan to nodes.
    
    Args:
        nodes: Original nodes
        fusion_plan: ML-predicted fusion plan
        
    Returns:
        Fused nodes
    """
    # Track which nodes have been fused
    fused_indices = set()
    fused_groups = []  # List of lists of node indices to fuse
    
    # Build fusion groups
    node_to_group = {}  # node_idx -> group_idx
    
    for idx, (i, j) in enumerate(fusion_plan.fusions):
        if i in node_to_group and j in node_to_group:
            # Both already in groups, merge groups
            group_i = node_to_group[i]
            group_j = node_to_group[j]
            if group_i != group_j:
                fused_groups[group_i].extend(fused_groups[group_j])
                for idx in fused_groups[group_j]:
                    node_to_group[idx] = group_i
                fused_groups[group_j] = []
        elif i in node_to_group:
            # i in group, add j
            group_idx = node_to_group[i]
            fused_groups[group_idx].append(j)
            node_to_group[j] = group_idx
        elif j in node_to_group:
            # j in group, add i
            group_idx = node_to_group[j]
            fused_groups[group_idx].append(i)
            node_to_group[i] = group_idx
        else:
            # Neither in group, create new group
            group_idx = len(fused_groups)
            fused_groups.append([i, j])
            node_to_group[i] = group_idx
            node_to_group[j] = group_idx
    
    # Remove empty groups
    fused_groups = [g for g in fused_groups if g]
    
    # Apply fusions
    result_nodes = []
    processed = set()
    
    for group in fused_groups:
        if not group:
            continue
        
        # Get nodes to fuse
        nodes_to_fuse = [nodes[idx] for idx in group]
        
        # Create fused node
        fused_node = self._create_fused_node(nodes_to_fuse)
        result_nodes.append(fused_node)
        
        processed.update(group)
    
    # Add unfused nodes
    for i, node in enumerate(nodes):
        if i not in processed:
            result_nodes.append(node)
    
    return result_nodes

def _create_fused_node(
    self,
    nodes: list[BaseSchedulerNode]
) -> BaseSchedulerNode:
    """Create a fused node from a list of nodes."""
    if len(nodes) == 1:
        return nodes[0]
    
    # Use existing FusedSchedulerNode
    return FusedSchedulerNode(self, nodes)

def _validate_fusion_plan(
    self,
    fusion_plan: FusionPlan,
    nodes: list[BaseSchedulerNode]
) -> tuple[bool, str]:
    """Validate fusion plan for correctness."""
    # Check indices are valid
    for i, j in fusion_plan.fusions:
        if i < 0 or i >= len(nodes) or j < 0 or j >= len(nodes):
            return False, f"Invalid node index: ({i}, {j})"
    
    # Check no self-loops
    for i, j in fusion_plan.fusions:
        if i == j:
            return False, f"Self-loop detected: ({i}, {i})"
    
    # Additional validation could include:
    # - Device compatibility
    # - Memory budget
    # - Dependency cycles
    
    return True, ""

def _ml_fuse_nodes_shadow(self, nodes: list[BaseSchedulerNode]):
    """Run ML scheduler in shadow mode (for data collection)."""
    try:
        fusion_plan = self.ml_orchestrator.predict_fusion_plan(
            nodes=nodes,
            device=self.current_device
        )
        
        # Record prediction but don't use it
        log.debug(f"Shadow mode: ML predicted {len(fusion_plan.fusions)} fusions")
        
    except Exception as e:
        log.debug(f"Shadow mode ML prediction failed: {e}")
```

### 2. Configuration (config.py)

Add ML scheduler configuration options:

```python
# File: torch/_inductor/config.py

# ML Scheduler Settings
ml_scheduler_enabled = os.environ.get("TORCHINDUCTOR_ML_SCHEDULER", "0") == "1"
ml_scheduler_mode = os.environ.get("TORCHINDUCTOR_ML_SCHEDULER_MODE", "hybrid")
ml_scheduler_model_path = os.environ.get("TORCHINDUCTOR_ML_SCHEDULER_MODEL_PATH", None)
ml_scheduler_confidence_threshold = float(
    os.environ.get("TORCHINDUCTOR_ML_SCHEDULER_CONFIDENCE", "0.75")
)
ml_scheduler_min_nodes = int(os.environ.get("TORCHINDUCTOR_ML_SCHEDULER_MIN_NODES", "5"))
ml_scheduler_max_nodes = int(os.environ.get("TORCHINDUCTOR_ML_SCHEDULER_MAX_NODES", "1000"))
ml_scheduler_cache_predictions = True
ml_scheduler_collect_feedback = True
ml_scheduler_fallback_on_error = True
```

### 3. Testing Integration

Create test file: `test/inductor/test_ml_scheduler.py`

```python
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor import config

class TestMLScheduler(TestCase):
    def test_ml_scheduler_basic(self):
        """Test basic ML scheduler functionality."""
        # Enable ML scheduler
        with config.patch(ml_scheduler_enabled=True, ml_scheduler_mode="shadow"):
            model = torch.nn.Linear(10, 10)
            compiled = torch.compile(model)
            
            x = torch.randn(5, 10)
            output = compiled(x)
            
            self.assertEqual(output.shape, (5, 10))
    
    def test_ml_scheduler_fallback(self):
        """Test fallback to heuristics."""
        with config.patch(
            ml_scheduler_enabled=True,
            ml_scheduler_mode="full",
            ml_scheduler_model_path=None  # No model -> fallback
        ):
            model = torch.nn.Linear(10, 10)
            compiled = torch.compile(model)
            
            x = torch.randn(5, 10)
            output = compiled(x)
            
            self.assertEqual(output.shape, (5, 10))

if __name__ == "__main__":
    run_tests()
```

## Usage Examples

### Enable ML Scheduler

```bash
# Enable in shadow mode (doesn't affect results)
export TORCHINDUCTOR_ML_SCHEDULER=1
export TORCHINDUCTOR_ML_SCHEDULER_MODE=shadow

# Enable in hybrid mode (uses ML with heuristic fallback)
export TORCHINDUCTOR_ML_SCHEDULER=1
export TORCHINDUCTOR_ML_SCHEDULER_MODE=hybrid
export TORCHINDUCTOR_ML_SCHEDULER_MODEL_PATH=/path/to/model.pt

# Enable in full mode (uses ML exclusively)
export TORCHINDUCTOR_ML_SCHEDULER=1
export TORCHINDUCTOR_ML_SCHEDULER_MODE=full
export TORCHINDUCTOR_ML_SCHEDULER_MODEL_PATH=/path/to/model.pt
```

### Python API

```python
import torch
from torch._inductor import config

# Configure ML scheduler
config.ml_scheduler_enabled = True
config.ml_scheduler_mode = "hybrid"
config.ml_scheduler_model_path = "/path/to/model.pt"

# Compile model
model = torch.nn.Linear(10, 10)
compiled_model = torch.compile(model)

# Use compiled model
x = torch.randn(5, 10)
output = compiled_model(x)
```

## Monitoring

Add monitoring hooks to track ML scheduler performance:

```python
# File: torch/_inductor/ml_scheduler/monitoring.py

class MLSchedulerMonitor:
    def __init__(self):
        self.metrics = {
            'total_compilations': 0,
            'ml_used': 0,
            'heuristic_fallback': 0,
            'errors': 0,
        }
    
    def record_compilation(self, used_ml, had_error):
        self.metrics['total_compilations'] += 1
        if used_ml:
            self.metrics['ml_used'] += 1
        else:
            self.metrics['heuristic_fallback'] += 1
        if had_error:
            self.metrics['errors'] += 1
    
    def get_stats(self):
        total = self.metrics['total_compilations']
        if total == 0:
            return {}
        
        return {
            'ml_usage_rate': self.metrics['ml_used'] / total,
            'fallback_rate': self.metrics['heuristic_fallback'] / total,
            'error_rate': self.metrics['errors'] / total,
        }
```

## Next Steps

1. Complete feature extractor implementations
2. Train initial models on benchmark data
3. Test in shadow mode
4. Gradual rollout with monitoring
5. Iterate based on feedback

