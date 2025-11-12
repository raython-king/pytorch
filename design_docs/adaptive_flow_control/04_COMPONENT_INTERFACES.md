# Adaptive Flow Control - Component Interfaces and APIs

## Table of Contents

1. [Core Components](#core-components)
2. [Public API](#public-api)
3. [Internal Interfaces](#internal-interfaces)
4. [Data Structures](#data-structures)
5. [Event System](#event-system)

---

## Core Components

### 1. FlowController

**Central manager for adaptive flow control**

```python
class FlowController:
    """
    Central flow control manager.
    
    Coordinates all flow control operations and enforces policies.
    """
    
    def __init__(self, config: FlowControlConfig):
        """
        Initialize flow controller.
        
        Args:
            config: Flow control configuration
        """
        self.config = config
        self.bandwidth_monitor = BandwidthMonitor(config.monitoring_interval_ms)
        self.congestion_detector = CongestionDetector(config.congestion_threshold)
        self.traffic_shaper = TrafficShaper(config.shaping_policy)
        self.adaptive_scheduler = AdaptiveScheduler(config.scheduling_policy)
        self.qos_manager = QoSManager(config.qos_enabled)
        self.path_router = PathRouter(config.routing_policy)
        
        self._active_flows: Dict[str, FlowDescriptor] = {}
        self._flow_policies: Dict[QoSClass, FlowPolicy] = {}
        self._statistics = FlowControlStatistics()
    
    def register_flow(
        self,
        flow_id: str,
        src_device: torch.device,
        dst_device: torch.device,
        size_gb: float,
        qos_class: QoSClass = QoSClass.BEST_EFFORT,
        qos_contract: Optional[QoSContract] = None
    ) -> FlowRegistrationResult:
        """
        Register a new flow for transfer.
        
        Args:
            flow_id: Unique flow identifier
            src_device: Source device
            dst_device: Destination device
            size_gb: Transfer size in GB
            qos_class: QoS class for this flow
            qos_contract: Optional QoS contract with guarantees
            
        Returns:
            FlowRegistrationResult with admission decision
            
        Example:
            >>> controller = FlowController(config)
            >>> result = controller.register_flow(
            ...     flow_id="transfer_001",
            ...     src_device=torch.device("cuda:0"),
            ...     dst_device=torch.device("cuda:1"),
            ...     size_gb=10.0,
            ...     qos_class=QoSClass.GUARANTEED,
            ...     qos_contract=QoSContract(
            ...         min_bandwidth_gbps=5.0,
            ...         max_latency_ms=100.0
            ...     )
            ... )
            >>> if result.admitted:
            ...     print(f"Flow admitted with priority {result.priority}")
        """
        # Admission control
        admission_result = self.qos_manager.admit_flow(
            flow_id, qos_class, qos_contract
        )
        
        if not admission_result.admitted:
            return FlowRegistrationResult(
                admitted=False,
                reason=admission_result.reason
            )
        
        # Create flow descriptor
        flow = FlowDescriptor(
            flow_id=flow_id,
            src_device=src_device,
            dst_device=dst_device,
            size_gb=size_gb,
            qos_class=qos_class,
            qos_contract=qos_contract,
            priority=admission_result.priority,
            creation_time=time.time()
        )
        
        # Select path
        path = self.path_router.select_path(
            src_device, dst_device, size_gb, qos_class
        )
        flow.path = path
        
        # Register with traffic shaper
        self.traffic_shaper.register_flow(flow)
        
        # Store flow
        self._active_flows[flow_id] = flow
        
        return FlowRegistrationResult(
            admitted=True,
            flow_descriptor=flow,
            estimated_latency_ms=self._estimate_latency(flow)
        )
    
    def schedule_transfer(
        self,
        flow_id: str
    ) -> TransferScheduleResult:
        """
        Schedule a flow for immediate transfer.
        
        Args:
            flow_id: Flow to schedule
            
        Returns:
            TransferScheduleResult with scheduling decision
            
        Raises:
            ValueError: If flow_id not registered
        """
        if flow_id not in self._active_flows:
            raise ValueError(f"Flow {flow_id} not registered")
        
        flow = self._active_flows[flow_id]
        
        # Check congestion
        congestion_info = self.congestion_detector.check_path(flow.path)
        
        if congestion_info.level >= CongestionLevel.HIGH:
            # Consider rerouting or backpressure
            if self.config.enable_rerouting:
                new_path = self.path_router.find_alternative_path(
                    flow.src_device, flow.dst_device, avoid=congestion_info.links
                )
                if new_path:
                    flow.path = new_path
                    congestion_info = self.congestion_detector.check_path(new_path)
        
        # Apply traffic shaping
        shaping_result = self.traffic_shaper.can_send(flow_id)
        
        if not shaping_result.allowed:
            return TransferScheduleResult(
                allowed=False,
                reason="rate_limited",
                wait_time_ms=shaping_result.wait_time_ms
            )
        
        # Schedule transfer
        schedule_info = self.adaptive_scheduler.schedule(flow, congestion_info)
        
        return TransferScheduleResult(
            allowed=True,
            path=flow.path,
            allocated_bandwidth_gbps=schedule_info.bandwidth_gbps,
            priority=schedule_info.priority,
            stream_id=schedule_info.stream_id
        )
    
    def complete_transfer(
        self,
        flow_id: str,
        actual_latency_ms: float,
        actual_throughput_gbps: float,
        success: bool = True
    ) -> None:
        """
        Mark transfer as complete and update statistics.
        
        Args:
            flow_id: Completed flow
            actual_latency_ms: Actual transfer latency
            actual_throughput_gbps: Actual throughput achieved
            success: Whether transfer succeeded
        """
        if flow_id not in self._active_flows:
            return
        
        flow = self._active_flows[flow_id]
        
        # Record completion
        flow.completion_time = time.time()
        flow.actual_latency_ms = actual_latency_ms
        flow.actual_throughput_gbps = actual_throughput_gbps
        flow.success = success
        
        # Update QoS manager
        self.qos_manager.release_flow(flow_id)
        
        # Update traffic shaper
        self.traffic_shaper.complete_flow(flow_id)
        
        # Update statistics
        self._statistics.record_transfer(flow)
        
        # Update ML models (online learning)
        if self.config.enable_online_learning:
            self._update_ml_models(flow)
        
        # Move to completed flows
        del self._active_flows[flow_id]
    
    def get_statistics(self) -> FlowControlStatistics:
        """
        Get flow control statistics.
        
        Returns:
            FlowControlStatistics object
        """
        stats = FlowControlStatistics(
            active_flows=len(self._active_flows),
            completed_flows=self._statistics.completed_flows,
            total_bytes_transferred=self._statistics.total_bytes_transferred,
            aggregate_throughput_gbps=self._calculate_aggregate_throughput(),
            average_latency_ms=self._statistics.average_latency_ms,
            p99_latency_ms=self._statistics.p99_latency_ms,
            jain_fairness_index=self._calculate_fairness_index(),
            qos_sla_compliance=self.qos_manager.get_compliance_rate(),
            congestion_events=self.congestion_detector.get_event_count(),
            bandwidth_utilization=self.bandwidth_monitor.get_utilization()
        )
        
        return stats
    
    def update_policy(self, policy: FlowPolicy) -> None:
        """Update flow control policy dynamically."""
        self.config.policy = policy
        self.adaptive_scheduler.set_policy(policy.scheduling_policy)
        self.traffic_shaper.set_policy(policy.shaping_policy)
        self.congestion_detector.set_threshold(policy.congestion_threshold)
    
    def enable(self) -> None:
        """Enable flow control system."""
        self.bandwidth_monitor.start()
        self.congestion_detector.start()
        self._enabled = True
    
    def disable(self) -> None:
        """Disable flow control system."""
        self.bandwidth_monitor.stop()
        self.congestion_detector.stop()
        self._enabled = False
    
    # Private methods
    def _estimate_latency(self, flow: FlowDescriptor) -> float:
        """Estimate transfer latency using ML model."""
        if self.config.use_ml_prediction:
            features = self._extract_latency_features(flow)
            prediction = self.ml_models['latency'].predict(features)
            return prediction['p99']  # Use P99 for conservative estimate
        else:
            # Fallback: simple calculation
            return (flow.size_gb / flow.allocated_bandwidth_gbps) * 1000
    
    def _update_ml_models(self, flow: FlowDescriptor) -> None:
        """Update ML models with completed transfer data."""
        # Update bandwidth predictor
        # Update latency predictor
        # Update congestion predictor
        pass
    
    def _calculate_aggregate_throughput(self) -> float:
        """Calculate current aggregate throughput."""
        return sum(
            flow.current_throughput_gbps 
            for flow in self._active_flows.values()
        )
    
    def _calculate_fairness_index(self) -> float:
        """Calculate Jain's fairness index."""
        if not self._active_flows:
            return 1.0
        
        allocations = [flow.allocated_bandwidth_gbps for flow in self._active_flows.values()]
        n = len(allocations)
        sum_x = sum(allocations)
        sum_x_sq = sum(x**2 for x in allocations)
        
        if sum_x_sq == 0:
            return 1.0
        
        return (sum_x ** 2) / (n * sum_x_sq)
```

### 2. BandwidthMonitor

```python
class BandwidthMonitor:
    """
    Monitor bandwidth utilization in real-time.
    
    Tracks per-link, per-device, and system-wide bandwidth.
    """
    
    def __init__(self, monitoring_interval_ms: float = 100.0):
        """
        Initialize bandwidth monitor.
        
        Args:
            monitoring_interval_ms: Monitoring frequency
        """
        self.monitoring_interval_ms = monitoring_interval_ms
        self.link_stats: Dict[str, LinkBandwidthStats] = {}
        self.device_stats: Dict[torch.device, DeviceBandwidthStats] = {}
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def get_link_bandwidth(self, link_id: str) -> BandwidthMeasurement:
        """
        Get current bandwidth for a link.
        
        Args:
            link_id: Link identifier
            
        Returns:
            BandwidthMeasurement with current stats
        """
        if link_id not in self.link_stats:
            return BandwidthMeasurement(
                current_gbps=0.0,
                utilization=0.0,
                peak_gbps=0.0
            )
        
        stats = self.link_stats[link_id]
        
        return BandwidthMeasurement(
            current_gbps=stats.current_bandwidth_gbps,
            utilization=stats.utilization,
            peak_gbps=stats.peak_bandwidth_gbps,
            moving_average_1s=stats.ma_1s,
            moving_average_10s=stats.ma_10s,
            variance=stats.variance
        )
    
    def get_device_bandwidth(
        self,
        device: torch.device
    ) -> DeviceBandwidthMeasurement:
        """Get aggregate bandwidth for a device."""
        if device not in self.device_stats:
            return DeviceBandwidthMeasurement(
                inbound_gbps=0.0,
                outbound_gbps=0.0,
                total_gbps=0.0,
                utilization=0.0
            )
        
        stats = self.device_stats[device]
        
        return DeviceBandwidthMeasurement(
            inbound_gbps=stats.inbound_bandwidth_gbps,
            outbound_gbps=stats.outbound_bandwidth_gbps,
            total_gbps=stats.total_bandwidth_gbps,
            utilization=stats.utilization
        )
    
    def get_utilization(self) -> float:
        """Get system-wide bandwidth utilization."""
        if not self.link_stats:
            return 0.0
        
        return np.mean([
            stats.utilization 
            for stats in self.link_stats.values()
        ])
    
    def predict_bandwidth(
        self,
        link_id: str,
        horizon_ms: float = 100.0
    ) -> BandwidthPrediction:
        """
        Predict future bandwidth availability.
        
        Args:
            link_id: Link to predict
            horizon_ms: Prediction horizon in milliseconds
            
        Returns:
            BandwidthPrediction with forecasted values
        """
        # Use ML model or time-series forecasting
        pass
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            self._collect_measurements()
            time.sleep(self.monitoring_interval_ms / 1000.0)
    
    def _collect_measurements(self) -> None:
        """Collect bandwidth measurements."""
        # Measure actual bandwidth usage
        # Update statistics
        pass
```

### 3. CongestionDetector

```python
class CongestionDetector:
    """
    Detect and predict congestion events.
    
    Monitors queue depths, latency, and utilization to identify congestion.
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize congestion detector.
        
        Args:
            threshold: Utilization threshold for congestion (0-1)
        """
        self.threshold = threshold
        self.link_congestion: Dict[str, CongestionState] = {}
        self.congestion_history: Deque[CongestionEvent] = deque(maxlen=1000)
        
        # ML predictor
        self.predictor: Optional[CongestionPredictor] = None
    
    def check_link(self, link_id: str) -> CongestionInfo:
        """
        Check congestion status of a link.
        
        Args:
            link_id: Link to check
            
        Returns:
            CongestionInfo with current status
        """
        state = self.link_congestion.get(link_id, CongestionState())
        
        # Determine congestion level
        if state.utilization < 0.6:
            level = CongestionLevel.NONE
        elif state.utilization < 0.75:
            level = CongestionLevel.LOW
        elif state.utilization < 0.85:
            level = CongestionLevel.MODERATE
        elif state.utilization < 0.95:
            level = CongestionLevel.HIGH
        else:
            level = CongestionLevel.CRITICAL
        
        # Predict future congestion
        if self.predictor:
            prob, confidence = self.predictor.predict(state, self.congestion_history)
        else:
            prob, confidence = 0.0, 0.0
        
        return CongestionInfo(
            level=level,
            utilization=state.utilization,
            queue_depth=state.queue_depth,
            latency_inflation=state.latency_inflation,
            congestion_probability=prob,
            confidence=confidence
        )
    
    def check_path(self, path: List[Link]) -> PathCongestionInfo:
        """
        Check congestion status across a path.
        
        Args:
            path: List of links in path
            
        Returns:
            PathCongestionInfo with aggregate status
        """
        link_info = [self.check_link(link.id) for link in path]
        
        # Overall congestion is worst link
        max_level = max(info.level for info in link_info)
        
        # Identify congested links
        congested_links = [
            link for link, info in zip(path, link_info)
            if info.level >= CongestionLevel.MODERATE
        ]
        
        return PathCongestionInfo(
            level=max_level,
            congested_links=congested_links,
            bottleneck_utilization=max(info.utilization for info in link_info)
        )
    
    def predict_congestion(
        self,
        link_id: str,
        horizon_ms: float = 100.0
    ) -> CongestionPrediction:
        """
        Predict future congestion.
        
        Args:
            link_id: Link to predict
            horizon_ms: Prediction horizon
            
        Returns:
            CongestionPrediction
        """
        if not self.predictor:
            return CongestionPrediction(probability=0.0, confidence=0.0)
        
        state = self.link_congestion.get(link_id, CongestionState())
        prob, confidence = self.predictor.predict(state, self.congestion_history)
        
        return CongestionPrediction(
            probability=prob,
            confidence=confidence,
            expected_time_ms=horizon_ms if prob > 0.5 else None
        )
```

Continuing in next message with more component interfaces...
