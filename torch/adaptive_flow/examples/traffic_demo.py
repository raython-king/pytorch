"""
Demo: Intelligent Traffic Management System

This example demonstrates the complete traffic management system
with priority-based scheduling, congestion control, and adaptive
bandwidth allocation.
"""

import time
import random
from torch.adaptive_flow import (
    TrafficManager,
    DataFlow,
    Priority,
    CongestionDetector,
    CongestionController,
    CongestionMetrics,
    CongestionState,
    BandwidthAllocator,
    SJF_Scheduler,
    WFQ_Scheduler,
    EDF_Scheduler,
    FlowInfo,
)


def simulate_basic_traffic():
    """Simulate basic traffic management."""
    print("=" * 80)
    print("Basic Traffic Management Demo")
    print("=" * 80)

    # Create traffic manager
    manager = TrafficManager()

    # Configure network links
    manager.set_link_capacity("node1", "node2", 1e9)  # 1 GB/s
    manager.set_link_capacity("node2", "node3", 500e6)  # 500 MB/s

    print("\nSubmitting flows...")

    # Submit various flows
    flows = [
        DataFlow(
            priority=Priority.HIGH,
            flow_id="critical_1",
            source="node1",
            dest="node2",
            size=10 * 1024 * 1024,  # 10 MB
        ),
        DataFlow(
            priority=Priority.MEDIUM,
            flow_id="normal_1",
            source="node1",
            dest="node2",
            size=100 * 1024 * 1024,  # 100 MB
        ),
        DataFlow(
            priority=Priority.LOW,
            flow_id="background_1",
            source="node1",
            dest="node2",
            size=500 * 1024 * 1024,  # 500 MB
        ),
    ]

    for flow in flows:
        manager.submit_flow(flow)
        print(f"  Submitted {flow.flow_id}: {flow.size / 1e6:.1f} MB, priority={flow.priority}")

    # Schedule flows
    print("\nScheduling flows...")
    scheduled = manager.schedule_flows()
    print(f"  Scheduled {len(scheduled)} flows: {scheduled}")

    # Simulate progress
    print("\nSimulating progress...")
    for flow_id in scheduled:
        # Simulate partial transfer
        transfer_size = 5 * 1024 * 1024  # 5 MB
        manager.update_flow_progress(flow_id, transfer_size)

        status = manager.get_flow_status(flow_id)
        print(f"  {flow_id}: {status['progress']:.1%} complete")

    # Statistics
    print("\nStatistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    link_stats = manager.get_link_statistics("node1", "node2")
    print("\nLink Statistics (node1 -> node2):")
    for key, value in link_stats.items():
        print(f"  {key}: {value}")


def simulate_congestion_control():
    """Simulate congestion control."""
    print("\n" + "=" * 80)
    print("Congestion Control Demo")
    print("=" * 80)

    # Create congestion detector and controller
    detector = CongestionDetector(queue_threshold=100, loss_threshold=0.01)
    controller = CongestionController(algorithm="aimd")

    link_id = "link1"
    flow_id = "flow1"

    # Initialize flow
    controller.initialize_flow(flow_id, initial_rate=100e6)  # 100 MB/s

    print(f"\nInitial rate: {controller.get_flow_rate(flow_id) / 1e6:.2f} MB/s")

    # Simulate normal conditions
    print("\nSimulating normal conditions...")
    for i in range(5):
        metrics = CongestionMetrics(
            queue_length=50,
            packet_loss_rate=0.001,
            rtt=0.05,
            throughput=100e6,
        )
        detector.update_metrics(link_id, metrics)

        state = detector.get_congestion_state(link_id)
        new_rate = controller.update_rate(flow_id, metrics, state)

        print(f"  Step {i+1}: rate={new_rate / 1e6:.2f} MB/s, state={state.value}")

    # Simulate congestion
    print("\nSimulating congestion...")
    for i in range(5):
        metrics = CongestionMetrics(
            queue_length=150,
            packet_loss_rate=0.05,
            rtt=0.15,
            throughput=50e6,
        )
        detector.update_metrics(link_id, metrics)

        state = detector.get_congestion_state(link_id)
        new_rate = controller.update_rate(flow_id, metrics, state)

        print(f"  Step {i+1}: rate={new_rate / 1e6:.2f} MB/s, state={state.value}")


def simulate_scheduling_policies():
    """Simulate different scheduling policies."""
    print("\n" + "=" * 80)
    print("Scheduling Policies Demo")
    print("=" * 80)

    # Create flows
    current_time = time.time()
    flows = [
        FlowInfo(flow_id="f1", size=10e6, priority=2, estimated_duration=10.0),
        FlowInfo(flow_id="f2", size=5e6, priority=1, estimated_duration=5.0),
        FlowInfo(flow_id="f3", size=20e6, priority=2, estimated_duration=20.0,
                deadline=current_time + 60),
        FlowInfo(flow_id="f4", size=8e6, priority=1, estimated_duration=8.0,
                deadline=current_time + 30),
    ]

    # Test SJF
    print("\nShortest Job First (SJF):")
    sjf = SJF_Scheduler(preemptive=False)
    selected = sjf.schedule(flows, available_slots=2)
    print(f"  Selected flows: {selected}")

    # Test WFQ
    print("\nWeighted Fair Queuing (WFQ):")
    wfq = WFQ_Scheduler()
    flows_with_weights = [
        FlowInfo(flow_id="f1", size=10e6, weight=1.0),
        FlowInfo(flow_id="f2", size=10e6, weight=2.0),
        FlowInfo(flow_id="f3", size=10e6, weight=1.0),
    ]
    selected = wfq.schedule(flows_with_weights, available_slots=2)
    print(f"  Selected flows: {selected}")

    # Test EDF
    print("\nEarliest Deadline First (EDF):")
    edf = EDF_Scheduler()
    selected = edf.schedule(flows, available_slots=2)
    print(f"  Selected flows: {selected}")


def simulate_bandwidth_allocation():
    """Simulate bandwidth allocation."""
    print("\n" + "=" * 80)
    print("Bandwidth Allocation Demo")
    print("=" * 80)

    allocator = BandwidthAllocator()

    # Set link capacity
    link_id = "link1"
    capacity = 1e9  # 1 GB/s
    allocator.set_link_capacity(link_id, capacity)

    print(f"\nLink capacity: {capacity / 1e9:.2f} GB/s")

    # Add flows with different demands
    print("\nAdding flows:")
    flows = [
        ("flow1", 400e6),  # 400 MB/s demand
        ("flow2", 400e6),  # 400 MB/s demand
        ("flow3", 400e6),  # 400 MB/s demand
        ("flow4", 400e6),  # 400 MB/s demand
    ]

    for flow_id, demand in flows:
        allocation = allocator.add_flow(flow_id, link_id, demand)
        print(f"  {flow_id}: demand={demand / 1e6:.2f} MB/s, "
              f"allocated={allocation / 1e6:.2f} MB/s")

    # Show link statistics
    print("\nLink statistics:")
    stats = allocator.get_link_statistics(link_id)
    for key, value in stats.items():
        if isinstance(value, float) and value > 1e6:
            print(f"  {key}: {value / 1e6:.2f} MB/s")
        else:
            print(f"  {key}: {value}")

    # Remove a flow
    print("\nRemoving flow1...")
    allocator.remove_flow("flow1")

    print("\nUpdated allocations:")
    for flow_id, _ in flows[1:]:
        allocation = allocator.get_allocation(flow_id)
        print(f"  {flow_id}: {allocation / 1e6:.2f} MB/s")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("ADAPTIVE FLOW CONTROL DEMONSTRATION")
    print("=" * 80)

    try:
        simulate_basic_traffic()
        simulate_congestion_control()
        simulate_scheduling_policies()
        simulate_bandwidth_allocation()

        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
