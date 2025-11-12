# Adaptive Flow Control - Algorithms and Mechanisms

## Table of Contents

1. [Bandwidth Allocation](#bandwidth-allocation)
2. [Congestion Control](#congestion-control)
3. [Traffic Shaping](#traffic-shaping)
4. [Flow Scheduling](#flow-scheduling)
5. [Multi-Path Routing](#multi-path-routing)
6. [QoS Enforcement](#qos-enforcement)

---

## Bandwidth Allocation

### 1. Max-Min Fair Allocation

**Algorithm:** Iterative water-filling for fair bandwidth distribution

```python
def max_min_fair_allocation(flows, total_bandwidth):
    """
    Allocate bandwidth with max-min fairness.
    
    Max-min fairness: Maximize the minimum allocation,
    then maximize the second minimum, and so on.
    
    Args:
        flows: List of flows with demands
        total_bandwidth: Total available bandwidth
        
    Returns:
        Dict mapping flow_id to allocated bandwidth
    """
    allocations = {}
    remaining_bandwidth = total_bandwidth
    remaining_flows = set(flow.id for flow in flows)
    
    while remaining_flows and remaining_bandwidth > 0:
        # Fair share for remaining flows
        fair_share = remaining_bandwidth / len(remaining_flows)
        
        # Find flows that need less than fair share
        satisfied = []
        for flow_id in remaining_flows:
            flow = flows[flow_id]
            if flow.demand <= fair_share:
                allocations[flow_id] = flow.demand
                remaining_bandwidth -= flow.demand
                satisfied.append(flow_id)
        
        if not satisfied:
            # All remaining flows get fair share
            for flow_id in remaining_flows:
                allocations[flow_id] = fair_share
            break
        
        # Remove satisfied flows
        for flow_id in satisfied:
            remaining_flows.remove(flow_id)
    
    return allocations


# Example usage
flows = [
    Flow(id=1, demand=10),  # Needs 10 GB/s
    Flow(id=2, demand=50),  # Needs 50 GB/s
    Flow(id=3, demand=50),  # Needs 50 GB/s
]

allocations = max_min_fair_allocation(flows, total_bandwidth=80)
# Result: {1: 10, 2: 35, 3: 35}
# Flow 1 gets full demand, flows 2 and 3 share remaining fairly
```

### 2. Weighted Fair Allocation

**Algorithm:** Weighted max-min fairness with priorities

```python
def weighted_fair_allocation(flows, total_bandwidth):
    """
    Allocate bandwidth with weighted fairness based on priorities.
    
    Args:
        flows: List of flows with demands and weights
        total_bandwidth: Total available bandwidth
        
    Returns:
        Dict mapping flow_id to allocated bandwidth
    """
    allocations = {}
    remaining_bandwidth = total_bandwidth
    remaining_flows = set(flow.id for flow in flows)
    
    while remaining_flows and remaining_bandwidth > 0:
        # Total weight of remaining flows
        total_weight = sum(flows[fid].weight for fid in remaining_flows)
        
        # Weighted fair share
        satisfied = []
        for flow_id in remaining_flows:
            flow = flows[flow_id]
            weighted_share = (flow.weight / total_weight) * remaining_bandwidth
            
            if flow.demand <= weighted_share:
                allocations[flow_id] = flow.demand
                remaining_bandwidth -= flow.demand
                satisfied.append(flow_id)
        
        if not satisfied:
            # Allocate weighted shares
            for flow_id in remaining_flows:
                flow = flows[flow_id]
                allocations[flow_id] = (flow.weight / total_weight) * remaining_bandwidth
            break
        
        # Remove satisfied flows
        for flow_id in satisfied:
            remaining_flows.remove(flow_id)
    
    return allocations
```

### 3. Hierarchical Bandwidth Allocation

**Three-Level Hierarchy:**
```
System Level
    ├── Device 0 (40 GB/s)
    │   ├── Critical Flows (20 GB/s)
    │   ├── Normal Flows (15 GB/s)
    │   └── Low Priority (5 GB/s)
    └── Device 1 (40 GB/s)
        ├── Critical Flows (10 GB/s)
        ├── Normal Flows (25 GB/s)
        └── Low Priority (5 GB/s)
```

```python
class HierarchicalBandwidthAllocator:
    """Hierarchical bandwidth allocation with QoS classes."""
    
    def __init__(self, system_bandwidth):
        self.system_bandwidth = system_bandwidth
        self.device_allocations = {}
        self.class_allocations = {}
        self.flow_allocations = {}
    
    def allocate(self, devices, flows):
        """Three-level allocation: System -> Device -> Class -> Flow"""
        
        # Level 1: Allocate to devices
        device_bandwidth = self._allocate_to_devices(devices)
        
        # Level 2: Allocate to QoS classes per device
        for device_id, bandwidth in device_bandwidth.items():
            device_flows = [f for f in flows if f.device == device_id]
            class_bandwidth = self._allocate_to_classes(
                device_flows, bandwidth
            )
            self.class_allocations[device_id] = class_bandwidth
            
            # Level 3: Allocate to flows within each class
            for qos_class, class_bw in class_bandwidth.items():
                class_flows = [f for f in device_flows 
                              if f.qos_class == qos_class]
                flow_bandwidth = max_min_fair_allocation(
                    class_flows, class_bw
                )
                self.flow_allocations.update(flow_bandwidth)
        
        return self.flow_allocations
    
    def _allocate_to_devices(self, devices):
        """Allocate bandwidth to devices based on load"""
        total_demand = sum(d.demand for d in devices)
        
        allocations = {}
        for device in devices:
            # Proportional to demand, up to device capacity
            allocation = min(
                device.capacity,
                (device.demand / total_demand) * self.system_bandwidth
            )
            allocations[device.id] = allocation
        
        return allocations
    
    def _allocate_to_classes(self, flows, total_bandwidth):
        """Allocate bandwidth to QoS classes"""
        # Reserve guaranteed bandwidth for critical classes
        reservations = {
            QoSClass.CRITICAL: 0.3 * total_bandwidth,  # 30% reserved
            QoSClass.HIGH: 0.2 * total_bandwidth,      # 20% reserved
            QoSClass.NORMAL: 0.0,                       # No reservation
            QoSClass.LOW: 0.0                          # No reservation
        }
        
        # Count flows per class
        class_flows = defaultdict(list)
        for flow in flows:
            class_flows[flow.qos_class].append(flow)
        
        # Allocate reserved bandwidth
        allocations = {}
        remaining = total_bandwidth
        
        for qos_class in [QoSClass.CRITICAL, QoSClass.HIGH]:
            if class_flows[qos_class]:
                allocations[qos_class] = reservations[qos_class]
                remaining -= reservations[qos_class]
        
        # Distribute remaining bandwidth fairly
        remaining_classes = [QoSClass.NORMAL, QoSClass.LOW]
        remaining_flow_count = sum(
            len(class_flows[c]) for c in remaining_classes
        )
        
        if remaining_flow_count > 0:
            for qos_class in remaining_classes:
                if class_flows[qos_class]:
                    share = (len(class_flows[qos_class]) / 
                            remaining_flow_count) * remaining
                    allocations[qos_class] = share
        
        return allocations
```

---

## Congestion Control

### 1. TCP-like Additive Increase Multiplicative Decrease (AIMD)

**Algorithm:** Gradual increase, rapid decrease on congestion

```python
class AIMDCongestionControl:
    """TCP-like AIMD congestion control."""
    
    def __init__(self, initial_rate, max_rate, min_rate):
        self.rate = initial_rate
        self.max_rate = max_rate
        self.min_rate = min_rate
        
        # AIMD parameters
        self.alpha = 1.0  # Additive increase (GB/s per RTT)
        self.beta = 0.5   # Multiplicative decrease factor
        
        self.ssthresh = max_rate / 2  # Slow start threshold
        self.in_slow_start = True
    
    def on_ack(self, rtt_ms):
        """Called when transfer completes successfully."""
        if self.in_slow_start:
            # Slow start: exponential increase
            self.rate = min(self.rate * 2, self.ssthresh)
            
            if self.rate >= self.ssthresh:
                self.in_slow_start = False
        else:
            # Congestion avoidance: additive increase
            self.rate = min(self.rate + self.alpha, self.max_rate)
    
    def on_congestion(self):
        """Called when congestion is detected."""
        # Multiplicative decrease
        self.ssthresh = self.rate / 2
        self.rate = max(self.rate * self.beta, self.min_rate)
        self.in_slow_start = False
    
    def get_rate(self):
        """Get current sending rate."""
        return self.rate


# Example usage
cc = AIMDCongestionControl(
    initial_rate=10.0,   # Start at 10 GB/s
    max_rate=100.0,      # Max 100 GB/s
    min_rate=1.0         # Min 1 GB/s
)

# Successful transfers -> increase rate
for _ in range(10):
    cc.on_ack(rtt_ms=1.0)
    print(f"Rate: {cc.get_rate():.2f} GB/s")

# Congestion detected -> decrease rate
cc.on_congestion()
print(f"After congestion: {cc.get_rate():.2f} GB/s")
```

### 2. Explicit Congestion Notification (ECN)

**Algorithm:** Proactive congestion signaling

```python
class ECNCongestionControl:
    """Explicit Congestion Notification based control."""
    
    def __init__(self, link_capacity):
        self.link_capacity = link_capacity
        self.current_rate = link_capacity * 0.5
        
        # ECN thresholds
        self.mark_threshold = 0.7    # Start marking at 70% util
        self.drop_threshold = 0.95   # Drop at 95% util
        
        # Rate adjustment
        self.increase_step = link_capacity * 0.05  # 5% increase
        self.decrease_factor = 0.8  # 20% decrease on ECN mark
    
    def check_congestion(self, utilization):
        """
        Check link utilization and return congestion signal.
        
        Returns:
            'none', 'mark', or 'drop'
        """
        if utilization >= self.drop_threshold:
            return 'drop'
        elif utilization >= self.mark_threshold:
            # Probabilistic marking based on utilization
            mark_prob = (utilization - self.mark_threshold) / \
                       (self.drop_threshold - self.mark_threshold)
            if random.random() < mark_prob:
                return 'mark'
        return 'none'
    
    def on_feedback(self, signal):
        """Adjust rate based on ECN feedback."""
        if signal == 'drop':
            # Severe congestion
            self.current_rate *= 0.5
        elif signal == 'mark':
            # Mild congestion
            self.current_rate *= self.decrease_factor
        else:
            # No congestion, increase rate
            self.current_rate = min(
                self.current_rate + self.increase_step,
                self.link_capacity
            )
    
    def get_rate(self):
        return self.current_rate
```

### 3. Adaptive Rate Control with ML

**Algorithm:** ML-predicted optimal rate

```python
class MLAdaptiveRateControl:
    """ML-based adaptive rate control."""
    
    def __init__(self, ml_model, link_capacity):
        self.model = ml_model
        self.link_capacity = link_capacity
        self.current_rate = link_capacity * 0.5
        
        # Historical observations
        self.history = deque(maxlen=100)
        
    def predict_optimal_rate(self, system_state):
        """
        Use ML model to predict optimal sending rate.
        
        Args:
            system_state: Current system state features
            
        Returns:
            Predicted optimal rate
        """
        # Extract features
        features = self._extract_features(system_state)
        
        # Get ML prediction
        predicted_rate = self.model.predict(features)
        
        # Clamp to valid range
        return np.clip(predicted_rate, 0.1, self.link_capacity)
    
    def _extract_features(self, state):
        """Extract features for ML model."""
        return torch.tensor([
            state.utilization,
            state.queue_depth,
            state.latency_ms,
            state.loss_rate,
            len(state.active_flows),
            state.bandwidth_variance,
            # Historical features
            np.mean([h.throughput for h in self.history]) if self.history else 0,
            np.std([h.latency for h in self.history]) if self.history else 0,
        ], dtype=torch.float32)
    
    def update(self, observation):
        """Update with new observation for online learning."""
        self.history.append(observation)
        
        # Periodically retrain model
        if len(self.history) >= 100:
            self._update_model()
    
    def _update_model(self):
        """Online model update."""
        # Extract training data from history
        features = [self._extract_features(h.state) for h in self.history]
        targets = [h.optimal_rate for h in self.history]
        
        # Update model (online learning)
        self.model.partial_fit(features, targets)
```

---

## Traffic Shaping

### 1. Token Bucket Algorithm

**Algorithm:** Allow bursts up to bucket size, refill at constant rate

```python
class TokenBucket:
    """Token bucket traffic shaper."""
    
    def __init__(self, rate_gbps, bucket_size_gb):
        """
        Initialize token bucket.
        
        Args:
            rate_gbps: Token refill rate (GB/s)
            bucket_size_gb: Maximum bucket capacity (GB)
        """
        self.rate_gbps = rate_gbps
        self.bucket_size_gb = bucket_size_gb
        self.tokens = bucket_size_gb  # Start with full bucket
        self.last_update = time.time()
    
    def try_consume(self, size_gb):
        """
        Try to consume tokens for a transfer.
        
        Args:
            size_gb: Transfer size in GB
            
        Returns:
            (allowed, wait_time_ms)
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        new_tokens = elapsed * self.rate_gbps
        self.tokens = min(self.tokens + new_tokens, self.bucket_size_gb)
        self.last_update = now
        
        # Check if enough tokens available
        if self.tokens >= size_gb:
            self.tokens -= size_gb
            return (True, 0.0)
        else:
            # Calculate wait time
            tokens_needed = size_gb - self.tokens
            wait_time_s = tokens_needed / self.rate_gbps
            return (False, wait_time_s * 1000)  # Return in ms
    
    def consume_blocking(self, size_gb):
        """Consume tokens, waiting if necessary."""
        allowed, wait_ms = self.try_consume(size_gb)
        
        if not allowed:
            time.sleep(wait_ms / 1000)
            self.tokens = 0  # Consumed after waiting
        
        return wait_ms


# Example usage
shaper = TokenBucket(rate_gbps=10.0, bucket_size_gb=50.0)

# Large burst transfer (30 GB) - allowed due to bucket
allowed, wait = shaper.try_consume(30.0)
print(f"Large burst: allowed={allowed}, wait={wait:.2f}ms")

# Another large transfer - need to wait
allowed, wait = shaper.try_consume(30.0)
print(f"Second burst: allowed={allowed}, wait={wait:.2f}ms")
```

### 2. Leaky Bucket Algorithm

**Algorithm:** Smooth traffic at constant rate

```python
class LeakyBucket:
    """Leaky bucket traffic shaper (constant rate output)."""
    
    def __init__(self, rate_gbps, queue_size_gb):
        """
        Initialize leaky bucket.
        
        Args:
            rate_gbps: Constant output rate (GB/s)
            queue_size_gb: Maximum queue size (GB)
        """
        self.rate_gbps = rate_gbps
        self.queue_size_gb = queue_size_gb
        self.queue = deque()
        self.queue_bytes = 0
        self.last_drain = time.time()
    
    def enqueue(self, transfer):
        """
        Enqueue a transfer.
        
        Returns:
            True if accepted, False if queue full
        """
        self._drain_queue()
        
        if self.queue_bytes + transfer.size_gb > self.queue_size_gb:
            return False  # Queue full, drop
        
        self.queue.append(transfer)
        self.queue_bytes += transfer.size_gb
        return True
    
    def _drain_queue(self):
        """Drain queue at constant rate."""
        now = time.time()
        elapsed = now - self.last_drain
        
        # Amount that can be drained
        drain_capacity = elapsed * self.rate_gbps
        
        # Drain transfers
        while self.queue and drain_capacity > 0:
            transfer = self.queue[0]
            
            if transfer.size_gb <= drain_capacity:
                # Complete this transfer
                self.queue.popleft()
                self.queue_bytes -= transfer.size_gb
                drain_capacity -= transfer.size_gb
                transfer.complete()
            else:
                # Partial drain
                transfer.size_gb -= drain_capacity
                drain_capacity = 0
        
        self.last_drain = now
    
    def get_wait_time(self, size_gb):
        """Estimate wait time for a transfer."""
        self._drain_queue()
        
        # Wait for queue to drain + own transfer
        total_ahead = self.queue_bytes + size_gb
        wait_time_s = total_ahead / self.rate_gbps
        return wait_time_s * 1000  # Return in ms
```

### 3. Hierarchical Token Bucket (HTB)

**Algorithm:** Multi-level token buckets for QoS classes

```python
class HierarchicalTokenBucket:
    """Hierarchical token bucket for multi-level shaping."""
    
    def __init__(self, config):
        """
        Initialize HTB.
        
        config structure:
        {
            'root': {'rate': 100, 'ceil': 100},
            'children': {
                'critical': {'rate': 30, 'ceil': 50, 'priority': 0},
                'normal': {'rate': 50, 'ceil': 80, 'priority': 1},
                'low': {'rate': 20, 'ceil': 100, 'priority': 2},
            }
        }
        """
        self.root_bucket = TokenBucket(
            rate_gbps=config['root']['rate'],
            bucket_size_gb=config['root']['rate'] * 2
        )
        
        self.class_buckets = {}
        for class_name, class_config in config['children'].items():
            self.class_buckets[class_name] = {
                'bucket': TokenBucket(
                    rate_gbps=class_config['rate'],
                    bucket_size_gb=class_config['rate'] * 2
                ),
                'ceil': class_config['ceil'],
                'priority': class_config['priority']
            }
    
    def try_send(self, transfer, qos_class):
        """
        Try to send transfer through hierarchical buckets.
        
        Returns:
            (allowed, wait_time_ms)
        """
        # Check class bucket first
        class_info = self.class_buckets[qos_class]
        allowed, wait_class = class_info['bucket'].try_consume(transfer.size_gb)
        
        if not allowed:
            return (False, wait_class)
        
        # Check root bucket
        allowed, wait_root = self.root_bucket.try_consume(transfer.size_gb)
        
        if not allowed:
            # Refund class tokens
            class_info['bucket'].tokens += transfer.size_gb
            return (False, wait_root)
        
        return (True, 0.0)
    
    def borrow_bandwidth(self, qos_class, size_gb):
        """
        Try to borrow unused bandwidth from parent or siblings.
        
        Lower priority classes can borrow up to their ceil if
        higher priority classes are not using their allocation.
        """
        class_info = self.class_buckets[qos_class]
        
        # Can borrow up to ceil
        ceil_tokens = class_info['ceil'] * 2  # Convert to bucket size
        
        if class_info['bucket'].tokens + size_gb <= ceil_tokens:
            # Borrow from root
            allowed, _ = self.root_bucket.try_consume(size_gb)
            if allowed:
                # Temporarily increase class bucket
                class_info['bucket'].tokens += size_gb
                return True
        
        return False
```

---

## Flow Scheduling

### 1. Shortest Job First (SJF)

**Algorithm:** Minimize average completion time

```python
class ShortestJobFirst:
    """SJF scheduler for flow scheduling."""
    
    def __init__(self):
        self.flows = []
        self.size_estimator = MLFlowSizeEstimator()
    
    def add_flow(self, flow):
        """Add flow to scheduler."""
        # Estimate size if unknown
        if flow.size_gb is None:
            flow.estimated_size_gb = self.size_estimator.estimate(flow)
        else:
            flow.estimated_size_gb = flow.size_gb
        
        self.flows.append(flow)
        # Keep sorted by estimated size
        self.flows.sort(key=lambda f: f.estimated_size_gb)
    
    def get_next_flow(self):
        """Get next flow to schedule (shortest first)."""
        if not self.flows:
            return None
        
        return self.flows.pop(0)
    
    def update_estimate(self, flow, actual_size_gb):
        """Update size estimator with actual size."""
        self.size_estimator.update(flow, actual_size_gb)


class MLFlowSizeEstimator:
    """ML-based flow size estimation."""
    
    def __init__(self):
        self.model = SimpleMLModel()
        self.history = []
    
    def estimate(self, flow):
        """Estimate flow size based on features."""
        features = torch.tensor([
            hash(flow.op_type) % 1000,
            len(flow.tensor_shapes),
            np.prod(flow.tensor_shapes[0]) if flow.tensor_shapes else 0,
            flow.src_device_id,
            flow.dst_device_id,
        ], dtype=torch.float32)
        
        return self.model.predict(features)
    
    def update(self, flow, actual_size):
        """Update model with actual observation."""
        self.history.append((flow, actual_size))
        
        if len(self.history) >= 100:
            self._retrain()
    
    def _retrain(self):
        """Retrain model with historical data."""
        features = [self._get_features(f) for f, _ in self.history]
        targets = [size for _, size in self.history]
        self.model.fit(features, targets)
```

### 2. Weighted Fair Queuing (WFQ)

**Algorithm:** Fair bandwidth sharing with weights

```python
class WeightedFairQueuing:
    """WFQ scheduler for flow scheduling."""
    
    def __init__(self, total_bandwidth_gbps):
        self.total_bandwidth_gbps = total_bandwidth_gbps
        self.flows = {}
        self.virtual_time = 0.0
    
    def add_flow(self, flow):
        """Add flow to WFQ scheduler."""
        self.flows[flow.id] = {
            'flow': flow,
            'weight': flow.priority_weight,
            'virtual_start': self.virtual_time,
            'virtual_finish': self.virtual_time + (flow.size_gb / flow.priority_weight),
            'transmitted': 0.0
        }
    
    def get_next_packet(self, packet_size_gb):
        """
        Get next packet to transmit (flow with earliest virtual finish time).
        
        Returns:
            flow_id to transmit from
        """
        if not self.flows:
            return None
        
        # Find flow with earliest virtual finish time
        next_flow_id = min(
            self.flows.keys(),
            key=lambda fid: self.flows[fid]['virtual_finish']
        )
        
        flow_info = self.flows[next_flow_id]
        
        # Update virtual time
        self.virtual_time = flow_info['virtual_finish']
        
        # Update transmitted amount
        flow_info['transmitted'] += packet_size_gb
        
        # Update virtual finish time
        remaining = flow_info['flow'].size_gb - flow_info['transmitted']
        if remaining > 0:
            flow_info['virtual_finish'] = (
                self.virtual_time + (packet_size_gb / flow_info['weight'])
            )
        else:
            # Flow complete
            del self.flows[next_flow_id]
        
        return next_flow_id
    
    def get_bandwidth_allocation(self):
        """Get current bandwidth allocation for each flow."""
        if not self.flows:
            return {}
        
        total_weight = sum(f['weight'] for f in self.flows.values())
        
        allocations = {}
        for flow_id, flow_info in self.flows.items():
            # Bandwidth proportional to weight
            allocations[flow_id] = (
                (flow_info['weight'] / total_weight) * 
                self.total_bandwidth_gbps
            )
        
        return allocations
```

Continuing in next message...

### 3. Earliest Deadline First (EDF)

**Algorithm:** Deadline-aware scheduling

```python
class EarliestDeadlineFirst:
    """EDF scheduler for deadline-aware flow scheduling."""
    
    def __init__(self):
        self.flows = []  # Min-heap by deadline
        self.latency_predictor = MLLatencyPredictor()
    
    def add_flow(self, flow):
        """Add flow with deadline."""
        # Predict completion time
        predicted_latency_ms = self.latency_predictor.predict(flow)
        flow.predicted_completion = time.time() + (predicted_latency_ms / 1000)
        
        # Check if deadline can be met
        if flow.deadline and flow.predicted_completion > flow.deadline:
            # Admission control: reject if deadline cannot be met
            return False
        
        # Add to heap (sorted by deadline)
        heapq.heappush(self.flows, (flow.deadline or float('inf'), flow))
        return True
    
    def get_next_flow(self):
        """Get flow with earliest deadline."""
        if not self.flows:
            return None
        
        _, flow = heapq.heappop(self.flows)
        return flow
    
    def check_deadlines(self):
        """Check if any deadlines are at risk."""
        at_risk = []
        current_time = time.time()
        
        for deadline, flow in self.flows:
            if deadline != float('inf'):
                time_to_deadline = deadline - current_time
                if time_to_deadline < flow.predicted_completion:
                    at_risk.append(flow)
        
        return at_risk
```

---

## Multi-Path Routing

### 1. Load-Aware Path Selection

**Algorithm:** Choose path based on current load

```python
class LoadAwareRouter:
    """Load-aware multi-path routing."""
    
    def __init__(self, topology):
        """
        Initialize router with network topology.
        
        topology: Graph of devices and links
        """
        self.topology = topology
        self.link_utilization = {}  # Link -> utilization
        self.path_cache = {}  # (src, dst) -> List[Path]
    
    def select_path(self, src_device, dst_device, transfer_size_gb):
        """
        Select optimal path for transfer.
        
        Args:
            src_device: Source device
            dst_device: Destination device
            transfer_size_gb: Transfer size
            
        Returns:
            Path (sequence of links)
        """
        # Get all possible paths
        paths = self._get_paths(src_device, dst_device)
        
        if not paths:
            return None
        
        # Score each path
        best_path = None
        best_score = float('-inf')
        
        for path in paths:
            score = self._score_path(path, transfer_size_gb)
            if score > best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    def _get_paths(self, src, dst):
        """Get all paths from src to dst."""
        # Cache paths for efficiency
        cache_key = (src, dst)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Find paths using topology
        paths = []
        
        # Direct paths
        if self.topology.has_direct_link(src, dst):
            paths.append([self.topology.get_link(src, dst)])
        
        # Multi-hop paths
        for intermediate in self.topology.get_neighbors(src):
            if self.topology.has_direct_link(intermediate, dst):
                path = [
                    self.topology.get_link(src, intermediate),
                    self.topology.get_link(intermediate, dst)
                ]
                paths.append(path)
        
        self.path_cache[cache_key] = paths
        return paths
    
    def _score_path(self, path, transfer_size_gb):
        """
        Score a path based on multiple criteria.
        
        Criteria:
        - Available bandwidth
        - Latency
        - Number of hops
        - Link utilization
        """
        score = 0.0
        
        # 1. Available bandwidth (bottleneck link)
        min_bandwidth = min(link.capacity_gbps for link in path)
        avg_utilization = np.mean([
            self.link_utilization.get(link.id, 0.0) 
            for link in path
        ])
        available_bandwidth = min_bandwidth * (1.0 - avg_utilization)
        score += available_bandwidth * 10.0  # Weight: 10x
        
        # 2. Latency (sum of link latencies)
        total_latency_ms = sum(link.latency_ms for link in path)
        score -= total_latency_ms * 0.1  # Weight: -0.1x
        
        # 3. Number of hops (prefer fewer hops)
        score -= len(path) * 5.0  # Weight: -5x
        
        # 4. Congestion risk
        max_utilization = max(
            self.link_utilization.get(link.id, 0.0) 
            for link in path
        )
        if max_utilization > 0.8:
            score -= 100.0  # Heavy penalty for congested links
        
        # 5. Transfer size (prefer high-bandwidth paths for large transfers)
        if transfer_size_gb > 1.0:  # Large transfer
            score += min_bandwidth * 5.0
        
        return score
    
    def update_utilization(self, link_id, utilization):
        """Update link utilization."""
        self.link_utilization[link_id] = utilization


# Example usage
topology = NetworkTopology()
topology.add_link('cpu', 'gpu0', capacity_gbps=16, latency_ms=0.1)  # PCIe
topology.add_link('gpu0', 'gpu1', capacity_gbps=300, latency_ms=0.01)  # NVLink
topology.add_link('gpu0', 'gpu2', capacity_gbps=300, latency_ms=0.01)  # NVLink
topology.add_link('gpu2', 'gpu1', capacity_gbps=300, latency_ms=0.01)  # NVLink

router = LoadAwareRouter(topology)

# Select path for transfer
path = router.select_path('gpu0', 'gpu1', transfer_size_gb=10.0)
print(f"Selected path: {' -> '.join(link.id for link in path)}")
```

### 2. ML-Based Path Optimization

**Algorithm:** Learn optimal routing from experience

```python
class MLRoutingOptimizer:
    """ML-based routing optimization."""
    
    def __init__(self, topology):
        self.topology = topology
        self.model = RoutingMLModel()
        self.history = deque(maxlen=1000)
    
    def select_path(self, src, dst, transfer):
        """
        Use ML to select optimal path.
        
        Returns:
            Path with predicted best performance
        """
        # Get candidate paths
        paths = self._get_candidate_paths(src, dst)
        
        if not paths:
            return None
        
        # Extract features for each path
        best_path = None
        best_predicted_perf = float('-inf')
        
        for path in paths:
            features = self._extract_path_features(path, transfer)
            predicted_perf = self.model.predict(features)
            
            if predicted_perf > best_predicted_perf:
                best_predicted_perf = predicted_perf
                best_path = path
        
        return best_path
    
    def _extract_path_features(self, path, transfer):
        """Extract features for ML model."""
        features = []
        
        # Path characteristics
        features.append(len(path))  # Number of hops
        features.append(min(link.capacity for link in path))  # Bottleneck
        features.append(sum(link.latency for link in path))  # Total latency
        features.append(np.mean([link.utilization for link in path]))  # Avg util
        features.append(max(link.utilization for link in path))  # Max util
        
        # Transfer characteristics
        features.append(transfer.size_gb)
        features.append(transfer.priority)
        features.append(len(transfer.dependencies))
        
        # System state
        features.append(self.get_system_load())
        features.append(len(self.get_active_transfers()))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def record_outcome(self, path, transfer, performance):
        """Record routing decision and outcome for learning."""
        features = self._extract_path_features(path, transfer)
        self.history.append({
            'features': features,
            'performance': performance,  # Throughput, latency, etc.
            'timestamp': time.time()
        })
        
        # Periodically update model
        if len(self.history) >= 100:
            self._update_model()
    
    def _update_model(self):
        """Update ML model with recent history."""
        features = torch.stack([h['features'] for h in self.history])
        targets = torch.tensor([h['performance'] for h in self.history])
        
        self.model.partial_fit(features, targets)
```

### 3. Adaptive Multi-Path Load Balancing

**Algorithm:** Split large transfers across multiple paths

```python
class MultiPathLoadBalancer:
    """Split transfers across multiple paths."""
    
    def __init__(self, router):
        self.router = router
        self.min_split_size_gb = 1.0  # Only split transfers > 1GB
    
    def split_transfer(self, transfer):
        """
        Split transfer across multiple paths if beneficial.
        
        Returns:
            List of (path, size_gb) tuples
        """
        src = transfer.src_device
        dst = transfer.dst_device
        total_size = transfer.size_gb
        
        # Don't split small transfers
        if total_size < self.min_split_size_gb:
            path = self.router.select_path(src, dst, total_size)
            return [(path, total_size)]
        
        # Get multiple paths
        paths = self.router._get_paths(src, dst)
        
        if len(paths) <= 1:
            # Only one path available
            return [(paths[0], total_size)]
        
        # Decide whether to split
        if self._should_split(paths, total_size):
            return self._allocate_across_paths(paths, total_size)
        else:
            # Use best single path
            best_path = self.router.select_path(src, dst, total_size)
            return [(best_path, total_size)]
    
    def _should_split(self, paths, total_size):
        """Decide if splitting is beneficial."""
        # Split if:
        # 1. Transfer is large enough (> 1GB)
        # 2. Multiple paths have good capacity
        # 3. Predicted completion time improves
        
        if total_size < self.min_split_size_gb:
            return False
        
        # Count paths with > 50% available capacity
        good_paths = sum(
            1 for path in paths 
            if self._get_available_capacity(path) > 0.5 * self._get_capacity(path)
        )
        
        return good_paths >= 2
    
    def _allocate_across_paths(self, paths, total_size):
        """Allocate transfer size across paths based on capacity."""
        # Get available capacity for each path
        capacities = [self._get_available_capacity(p) for p in paths]
        total_capacity = sum(capacities)
        
        if total_capacity == 0:
            # Fallback: equal split
            split_size = total_size / len(paths)
            return [(path, split_size) for path in paths]
        
        # Proportional allocation
        allocations = []
        for path, capacity in zip(paths, capacities):
            size = (capacity / total_capacity) * total_size
            allocations.append((path, size))
        
        return allocations
    
    def _get_available_capacity(self, path):
        """Get available capacity on path (min of all links)."""
        return min(
            link.capacity_gbps * (1.0 - link.utilization)
            for link in path
        )
    
    def _get_capacity(self, path):
        """Get total capacity of path."""
        return min(link.capacity_gbps for link in path)
```

---

## QoS Enforcement

### 1. Admission Control

**Algorithm:** Accept/reject flows based on QoS guarantees

```python
class QoSAdmissionControl:
    """Admission control for QoS flows."""
    
    def __init__(self, total_bandwidth_gbps):
        self.total_bandwidth = total_bandwidth_gbps
        self.reserved_bandwidth = 0.0
        self.admitted_flows = {}
    
    def admit_flow(self, flow):
        """
        Decide whether to admit flow with QoS requirements.
        
        Returns:
            (admitted, reason)
        """
        if flow.qos_class == QoSClass.BEST_EFFORT:
            # Always admit best-effort flows
            return (True, "best_effort")
        
        # Check bandwidth availability
        required_bw = flow.min_bandwidth_gbps
        available_bw = self.total_bandwidth - self.reserved_bandwidth
        
        if required_bw > available_bw:
            return (False, "insufficient_bandwidth")
        
        # Check if we can meet latency guarantee
        predicted_latency = self._predict_latency(flow, required_bw)
        if predicted_latency > flow.max_latency_ms:
            return (False, "latency_guarantee_cannot_be_met")
        
        # Admit flow
        self.reserved_bandwidth += required_bw
        self.admitted_flows[flow.id] = flow
        
        return (True, "admitted")
    
    def release_flow(self, flow_id):
        """Release resources when flow completes."""
        if flow_id in self.admitted_flows:
            flow = self.admitted_flows[flow_id]
            self.reserved_bandwidth -= flow.min_bandwidth_gbps
            del self.admitted_flows[flow_id]
    
    def _predict_latency(self, flow, allocated_bandwidth):
        """Predict latency with allocated bandwidth."""
        # Simple model: latency = size / bandwidth + base_latency
        transfer_time_ms = (flow.size_gb / allocated_bandwidth) * 1000
        base_latency_ms = 1.0  # Base latency
        queue_latency_ms = self._estimate_queue_latency()
        
        return transfer_time_ms + base_latency_ms + queue_latency_ms
    
    def _estimate_queue_latency(self):
        """Estimate queuing latency."""
        # Based on number of flows and utilization
        num_flows = len(self.admitted_flows)
        utilization = self.reserved_bandwidth / self.total_bandwidth
        
        # Simple model: latency increases with utilization
        return num_flows * utilization * 2.0  # ms
```

### 2. Priority Enforcement

**Algorithm:** Ensure high-priority flows get resources

```python
class PriorityEnforcer:
    """Enforce priority-based QoS."""
    
    def __init__(self, bandwidth_allocator):
        self.allocator = bandwidth_allocator
        self.priority_queue = {
            0: deque(),  # Critical
            1: deque(),  # High
            2: deque(),  # Normal
            3: deque(),  # Low
        }
    
    def enqueue_transfer(self, transfer):
        """Enqueue transfer in priority queue."""
        priority = transfer.priority
        self.priority_queue[priority].append(transfer)
    
    def get_next_transfer(self):
        """Get next transfer respecting priorities."""
        # Try each priority level
        for priority in sorted(self.priority_queue.keys()):
            if self.priority_queue[priority]:
                transfer = self.priority_queue[priority].popleft()
                return transfer
        
        return None
    
    def preempt_if_needed(self, high_priority_transfer):
        """
        Preempt lower-priority transfers if needed for high-priority.
        
        Returns:
            List of preempted transfers
        """
        preempted = []
        
        if not high_priority_transfer.preemption_allowed:
            return preempted
        
        # Calculate bandwidth needed
        required_bw = high_priority_transfer.min_bandwidth_gbps
        available_bw = self.allocator.get_available_bandwidth()
        
        if available_bw >= required_bw:
            return preempted  # No preemption needed
        
        # Need to preempt lower-priority flows
        needed_bw = required_bw - available_bw
        
        # Try to preempt from lower priorities
        for priority in reversed(range(high_priority_transfer.priority + 1, 4)):
            if needed_bw <= 0:
                break
            
            for transfer in list(self.priority_queue[priority]):
                if needed_bw <= 0:
                    break
                
                # Preempt this transfer
                self.priority_queue[priority].remove(transfer)
                preempted.append(transfer)
                needed_bw -= transfer.allocated_bandwidth
        
        return preempted
```

### 3. SLA Monitoring

**Algorithm:** Monitor and enforce SLA compliance

```python
class SLAMonitor:
    """Monitor SLA compliance for QoS flows."""
    
    def __init__(self):
        self.flow_stats = {}
        self.violations = []
    
    def record_transfer(self, flow_id, latency_ms, throughput_gbps):
        """Record transfer performance."""
        if flow_id not in self.flow_stats:
            self.flow_stats[flow_id] = {
                'latencies': deque(maxlen=100),
                'throughputs': deque(maxlen=100),
                'violations': 0
            }
        
        stats = self.flow_stats[flow_id]
        stats['latencies'].append(latency_ms)
        stats['throughputs'].append(throughput_gbps)
    
    def check_sla(self, flow):
        """Check if flow is meeting SLA."""
        if flow.id not in self.flow_stats:
            return True  # No data yet
        
        stats = self.flow_stats[flow.id]
        
        violations = []
        
        # Check latency SLA
        if flow.max_latency_ms:
            p99_latency = np.percentile(stats['latencies'], 99)
            if p99_latency > flow.max_latency_ms:
                violations.append(f"P99 latency {p99_latency:.2f}ms > {flow.max_latency_ms}ms")
        
        # Check bandwidth SLA
        if flow.min_bandwidth_gbps:
            avg_throughput = np.mean(stats['throughputs'])
            if avg_throughput < flow.min_bandwidth_gbps:
                violations.append(f"Avg throughput {avg_throughput:.2f} < {flow.min_bandwidth_gbps}")
        
        if violations:
            self.record_violation(flow, violations)
            return False
        
        return True
    
    def record_violation(self, flow, violations):
        """Record SLA violation."""
        self.violations.append({
            'flow_id': flow.id,
            'timestamp': time.time(),
            'violations': violations
        })
        
        self.flow_stats[flow.id]['violations'] += 1
    
    def get_compliance_rate(self, flow_id=None):
        """Get SLA compliance rate."""
        if flow_id:
            if flow_id not in self.flow_stats:
                return 1.0
            stats = self.flow_stats[flow_id]
            total_samples = len(stats['latencies'])
            if total_samples == 0:
                return 1.0
            return 1.0 - (stats['violations'] / total_samples)
        else:
            # Overall compliance
            total_violations = sum(s['violations'] for s in self.flow_stats.values())
            total_samples = sum(len(s['latencies']) for s in self.flow_stats.values())
            if total_samples == 0:
                return 1.0
            return 1.0 - (total_violations / total_samples)
```

---

## Performance Analysis

### Theoretical Bounds

**Throughput:**
- Max aggregate throughput: `min(sum(link_capacities), sum(source_rates))`
- Per-flow throughput (max-min fairness): `min(demand, fair_share)`

**Latency:**
- Min latency: `transfer_size / link_capacity + propagation_delay`
- Queue latency: `queue_size / service_rate` (M/M/1 model)

**Fairness:**
- Jain's Fairness Index: `(sum(x_i))^2 / (n * sum(x_i^2))` where x_i is allocation
- Perfect fairness: JFI = 1.0

### Complexity Analysis

```
Algorithm Complexities:
├── Max-Min Fair Allocation: O(n^2) worst case, O(n) average
├── Token Bucket: O(1) per operation
├── WFQ Scheduling: O(log n) per packet
├── Path Selection: O(p * l) where p=paths, l=links per path
└── SLA Monitoring: O(1) per transfer, O(n) for aggregate
```

---

## Summary

This document covers the core algorithms for:
1. Fair and efficient bandwidth allocation
2. Congestion control and avoidance
3. Traffic shaping and rate limiting
4. Intelligent flow scheduling
5. Multi-path routing and load balancing
6. QoS enforcement and SLA monitoring

Next: See `03_ML_MODELS.md` for ML model architectures and training procedures.
