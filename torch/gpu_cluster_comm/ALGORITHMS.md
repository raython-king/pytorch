# Communication Algorithms
# 通讯算法详解

This document describes the collective communication algorithms implemented in the GPU cluster communication optimizer.

本文档描述GPU集群通讯优化器中实现的集合通讯算法。

## AllReduce Algorithms

### 1. Ring AllReduce

**Best for**: Large messages, high bandwidth utilization

**Time Complexity**: `T = 2(N-1) * (α + β * M/N)`
- N: number of ranks
- M: message size
- α: latency
- β: inverse bandwidth

**Algorithm**:
```
Phase 1: Reduce-Scatter
  For step = 0 to N-2:
    Each rank sends chunk i to rank (i+1) mod N
    Each rank reduces received chunk with local chunk

Phase 2: AllGather
  For step = 0 to N-2:
    Each rank sends chunk i to rank (i+1) mod N
    Each rank stores received chunk
```

**Advantages**:
- Bandwidth optimal: 2(N-1)/N efficiency
- Works with any number of ranks
- Good for large messages

**Disadvantages**:
- High latency for small messages
- 2(N-1) communication steps

### 2. Binary Tree AllReduce

**Best for**: Small messages, latency-sensitive operations

**Time Complexity**: `T = 2*log(N) * (α + β * M)`

**Algorithm**:
```
Phase 1: Reduce (Bottom-up)
  For level = log(N)-1 down to 0:
    Children send to parent
    Parent reduces with local data

Phase 2: Broadcast (Top-down)
  For level = 0 to log(N)-1:
    Parent sends to children
```

**Advantages**:
- Latency optimal: 2*log(N) steps
- Fewer hops than ring
- Good for small messages

**Disadvantages**:
- Not bandwidth optimal
- Root can be a bottleneck

### 3. Recursive Halving-Doubling

**Best for**: Power-of-2 ranks, medium messages

**Time Complexity**: `T = log(N) * (α + β * M/2)`

**Algorithm**:
```
Phase 1: Recursive Halving (Reduce-Scatter)
  For step = 0 to log(N)-1:
    Ranks exchange and reduce with distance 2^step partners

Phase 2: Recursive Doubling (AllGather)
  For step = log(N)-1 down to 0:
    Ranks exchange with distance 2^step partners
```

**Advantages**:
- Efficient for power-of-2 ranks
- Lower latency than ring
- Good bandwidth utilization

**Disadvantages**:
- Requires power-of-2 ranks (or padding)
- More complex than ring

### 4. Hierarchical AllReduce

**Best for**: Multi-node clusters with heterogeneous interconnects

**Time Complexity**:
```
T = T_intra_reduce + T_inter_allreduce + T_intra_broadcast
  = 2(N_local-1)*(α_nvlink + β_nvlink*M/N_local)
    + 2(N_node-1)*(α_ib + β_ib*M/N_node)
    + log(N_local)*(α_nvlink + β_nvlink*M)
```

**Algorithm**:
```
Phase 1: Intra-node Reduce
  Each node: Reduce to rank 0 using NVLink

Phase 2: Inter-node AllReduce
  Node leaders: AllReduce using InfiniBand

Phase 3: Intra-node Broadcast
  Each node: Broadcast from rank 0 using NVLink
```

**Advantages**:
- Exploits fast intra-node links (NVLink)
- Reduces inter-node traffic
- Scales to many nodes

**Disadvantages**:
- More complex
- Requires topology awareness

## Algorithm Selection Strategy

The `AdaptiveCollectiveOptimizer` selects algorithms based on:

### Message Size

```
Small (< 32 KB):
  - Use tree-based algorithms (minimize latency)
  - Best: Recursive Halving-Doubling (if power-of-2)
  - Fallback: Binary Tree

Medium (32 KB - 1 MB):
  - Compare algorithms and choose best
  - Consider: Ring, Tree, Halving-Doubling

Large (> 1 MB):
  - Use ring-based algorithms (maximize bandwidth)
  - Best: Ring AllReduce
  - Alternative: Hierarchical (for multi-node)
```

### Number of Ranks

```
Few ranks (≤ 4):
  - Use tree algorithms (fewer hops)

Medium ranks (5-16):
  - Use halving-doubling (if power-of-2)
  - Otherwise: Ring

Many ranks (> 16):
  - Use ring or hierarchical
  - Hierarchical preferred for multi-node
```

### Topology

```
Single Node (NVLink):
  - Use any algorithm
  - NVLink has high bandwidth, low latency

Multi-Node (InfiniBand):
  - Prefer hierarchical
  - Reduce inter-node traffic

Heterogeneous:
  - Use hierarchical
  - Optimize for each level separately
```

## Cost Model

### Ring AllReduce Cost

```python
def ring_cost(M, N, α, β):
    """
    M: message size (bytes)
    N: number of ranks
    α: latency (microseconds)
    β: inverse bandwidth (us/byte)
    """
    chunk_size = M / N
    steps = 2 * (N - 1)  # Reduce-scatter + AllGather
    return steps * (α + β * chunk_size)
```

### Tree AllReduce Cost

```python
def tree_cost(M, N, α, β):
    """Binary tree AllReduce"""
    depth = math.ceil(math.log2(N))
    steps = 2 * depth  # Up + Down
    return steps * (α + β * M)
```

### Halving-Doubling Cost

```python
def halving_doubling_cost(M, N, α, β):
    """Recursive halving-doubling"""
    steps = math.ceil(math.log2(N))
    avg_chunk = M / 2  # Average chunk size
    return 2 * steps * (α + β * avg_chunk)
```

### Hierarchical Cost

```python
def hierarchical_cost(M, N, N_local, N_node, α_intra, β_intra, α_inter, β_inter):
    """Two-level hierarchical"""
    # Intra-node reduce
    intra_reduce = 2 * (N_local - 1) * (α_intra + β_intra * M / N_local)

    # Inter-node allreduce
    inter_allreduce = 2 * (N_node - 1) * (α_inter + β_inter * M / N_node)

    # Intra-node broadcast
    intra_broadcast = math.log2(N_local) * (α_intra + β_intra * M)

    return intra_reduce + inter_allreduce + intra_broadcast
```

## Optimization Techniques

### 1. Pipelining

Break message into chunks and pipeline:

```
Without pipelining:
  Time = N_steps * (latency + transfer_time)

With pipelining (K chunks):
  Time = latency + (N_steps + K - 1) * transfer_time/K
  Speedup ≈ K (for large K)
```

### 2. Chunk Size Optimization

Optimal chunk size balances:
- Smaller chunks: Better pipelining
- Larger chunks: Lower overhead

```python
optimal_chunks = min(
    message_size / min_chunk_size,
    4 * num_ranks  # Heuristic
)
```

### 3. Topology-Aware Ring Order

Optimize ring order to follow high-bandwidth links:

```python
def optimal_ring_order(bandwidth_matrix):
    """Greedy nearest-neighbor"""
    visited = {0}
    order = [0]
    current = 0

    while len(visited) < N:
        # Find unvisited neighbor with highest bandwidth
        next_rank = max(
            (r for r in range(N) if r not in visited),
            key=lambda r: bandwidth_matrix[current, r]
        )
        order.append(next_rank)
        visited.add(next_rank)
        current = next_rank

    return order
```

## Example: Selecting Best Algorithm

```python
def select_algorithm(message_size, num_ranks, topology):
    """Algorithm selection logic"""

    # Get network parameters
    bandwidth, latency = get_network_params(topology)

    # Small messages: latency-bound
    if message_size < 32 * 1024:
        if is_power_of_two(num_ranks):
            return HALVING_DOUBLING
        else:
            return TREE

    # Large messages: bandwidth-bound
    elif message_size > 1 * 1024 * 1024:
        if topology.num_nodes > 1:
            return HIERARCHICAL
        else:
            return RING

    # Medium messages: compare costs
    else:
        costs = {
            RING: ring_cost(message_size, num_ranks, latency, 1/bandwidth),
            TREE: tree_cost(message_size, num_ranks, latency, 1/bandwidth),
        }

        if is_power_of_two(num_ranks):
            costs[HALVING_DOUBLING] = halving_doubling_cost(
                message_size, num_ranks, latency, 1/bandwidth
            )

        return min(costs, key=costs.get)
```

## Performance Comparison

### Bandwidth Efficiency

```
Ring AllReduce:        2(N-1)/N  ≈ 100% for large N
Tree AllReduce:        M/log(N)  ≈ 50-75%
Halving-Doubling:      ≈ 90%
Hierarchical:          ≈ 95%
```

### Latency Efficiency

```
Ring AllReduce:        O(N)
Tree AllReduce:        O(log N)   ← Best
Halving-Doubling:      O(log N)   ← Best
Hierarchical:          O(log N)
```

## References

1. Rabenseifner, R. (2004). "Optimization of Collective Reduction Operations"
2. Thakur, R. et al. (2005). "Optimization of Collective Communication Operations in MPICH"
3. Jeaugey, S. (2017). "NCCL 2.0" (Ring AllReduce implementation)
4. Patarasuk, P. & Yuan, X. (2009). "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"

## Implementation Notes

- All algorithms in `collective_optimizer.py`
- Cost models in `AlgorithmCostModel` class
- Selection logic in `AdaptiveCollectiveOptimizer.select_allreduce_algorithm()`
- Topology awareness in `TopologyManager`

## Future Improvements

1. **ML-based selection**: Train model on historical data
2. **Multi-algorithm hybrid**: Use different algorithms for different chunks
3. **Adaptive tuning**: Online optimization based on observed performance
4. **RDMA optimization**: Leverage RDMA for zero-copy transfers
5. **GPU-Direct**: Direct GPU-to-GPU transfers without CPU involvement
