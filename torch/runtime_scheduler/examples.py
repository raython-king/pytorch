"""
Examples for PyTorch Runtime Scheduler

This file contains practical examples demonstrating the monitoring,
profiling, and integration features.
"""

import time
import torch
import torch.nn as nn


# Example 1: Basic Monitoring
def example_basic_monitoring():
    """Example of basic performance monitoring."""
    from torch.runtime_scheduler import PerformanceMonitor, TimedOperation

    print("=== Example 1: Basic Monitoring ===")

    monitor = PerformanceMonitor()
    monitor.start()

    # Simulate some operations
    for i in range(10):
        with TimedOperation(monitor, f"op_{i}", "cuda:0"):
            time.sleep(0.001 * (i + 1))

    # Get and print summary
    summary = monitor.get_summary()
    print(f"Total operations: {summary['total_operations']}")
    print(f"Operation counts: {summary['operation_counts']}")

    monitor.stop()
    print()


# Example 2: Profiling with Analysis
def example_profiling():
    """Example of profiling with decision analysis."""
    from torch.runtime_scheduler import RuntimeSchedulerProfiler

    print("=== Example 2: Profiling ===")

    profiler = RuntimeSchedulerProfiler(enabled=True)
    profiler.start()

    # Simulate scheduling decisions
    for i in range(5):
        decision_id = profiler.record_decision(
            operation=f"op_{i}",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0", "cuda:1"],
            decision_factors={"latency": 0.7, "memory": 0.3},
            predicted_latency=0.001
        )

        # Simulate execution
        time.sleep(0.001)
        actual_latency = 0.0011 + i * 0.0001

        profiler.update_decision_outcome(decision_id, actual_latency)

    profiler.stop()

    # Analyze effectiveness
    analysis = profiler.analyze_scheduling_effectiveness()
    print(f"Total decisions: {analysis['total_decisions']}")
    print(f"Mean prediction error: {analysis.get('mean_prediction_error', 0):.2%}")

    # Export trace
    profiler.export_chrome_trace("example_trace.json")
    print("Trace exported to example_trace.json")
    print()


# Example 3: Configuration and Control
def example_configuration():
    """Example of runtime scheduler configuration."""
    from torch.runtime_scheduler import (
        enable_runtime_scheduler,
        SchedulerContext
    )

    print("=== Example 3: Configuration ===")

    # Enable globally
    scheduler = enable_runtime_scheduler(
        mode='heuristic',
        target='latency',
        devices=['cuda:0'] if torch.cuda.is_available() else ['cpu'],
        monitoring=True
    )

    print(f"Scheduler enabled with mode: {scheduler.config.mode}")

    # Use context manager for temporary config
    with SchedulerContext(mode='ml', target='throughput'):
        print("Inside context: ML mode")

    # Get statistics
    stats = scheduler.get_stats()
    print(f"Operations scheduled: {stats['operations_scheduled']}")
    print()


# Example 4: Data Collection
def example_data_collection():
    """Example of collecting execution traces."""
    from torch.runtime_scheduler.training import RuntimeDataCollector

    print("=== Example 4: Data Collection ===")

    collector = RuntimeDataCollector(
        max_traces=100,
        sampling_rate=1.0,
        auto_save=False
    )

    # Collect traces
    for i in range(20):
        collector.collect(
            operation=f"op_{i % 5}",
            input_shapes=[(2, 3, 224, 224)],
            input_dtypes=["float32"],
            device="cuda:0",
            actual_latency=0.001 * (i + 1),
            memory_allocated=1024 * (i + 1)
        )

    # Get statistics
    stats = collector.get_statistics()
    print(f"Traces collected: {stats['total_collected']}")

    # Create training examples
    examples = collector.create_training_examples()
    print(f"Training examples created: {len(examples)}")
    print()


# Example 5: Integration Hooks
def example_hooks():
    """Example of PyTorch integration hooks."""
    from torch.runtime_scheduler.integration import (
        RuntimeSchedulerHooks,
        HookMode,
        HookContext
    )

    print("=== Example 5: Integration Hooks ===")

    hooks = RuntimeSchedulerHooks(mode=HookMode.SHADOW)

    # Register callback
    operations_seen = []

    def operation_callback(op_info):
        operations_seen.append(op_info.op_name)
        return op_info

    hooks.register_operation_dispatch_callback(operation_callback)

    # Use with context manager
    with HookContext(mode=HookMode.SHADOW):
        # Hooks active here
        print("Hooks enabled in shadow mode")

    # Get statistics
    stats = hooks.get_stats()
    print(f"Operations intercepted: {stats['operations_intercepted']}")
    print()


# Example 6: Complete Workflow
def example_complete_workflow():
    """Example of complete monitoring and profiling workflow."""
    from torch.runtime_scheduler import (
        PerformanceMonitor,
        RuntimeSchedulerProfiler,
        Visualizer,
        Alerter
    )

    print("=== Example 6: Complete Workflow ===")

    # Setup components
    monitor = PerformanceMonitor()
    profiler = RuntimeSchedulerProfiler(enabled=True)
    visualizer = Visualizer(monitor, log_dir=None)

    def alert_callback(metric, info):
        print(f"ALERT: {metric} = {info['value']:.3f}")

    alerter = Alerter(monitor, alert_callback=alert_callback)
    alerter.add_threshold("latency", 5.0, ">")

    # Start monitoring and profiling
    monitor.start()
    profiler.start()

    # Simulate workload
    for i in range(5):
        # Record in monitor
        monitor.record_operation(
            op_name="forward",
            device="cuda:0",
            latency=0.001 * (i + 1)
        )

        # Record in profiler
        decision_id = profiler.record_decision(
            operation="forward",
            chosen_device="cuda:0",
            candidate_devices=["cuda:0", "cuda:1"],
            decision_factors={"latency": 0.8},
            predicted_latency=0.001 * (i + 1)
        )

        profiler.update_decision_outcome(
            decision_id,
            actual_latency=0.001 * (i + 1)
        )

    # Check alerts
    alerts = alerter.check_alerts()

    # Print dashboard
    visualizer.print_dashboard()

    # Get analysis
    analysis = profiler.analyze_scheduling_effectiveness()
    print(f"\nScheduling analysis:")
    print(f"  Total decisions: {analysis['total_decisions']}")

    # Cleanup
    profiler.stop()
    monitor.stop()
    visualizer.close()
    print()


def main():
    """Run all examples."""
    print("PyTorch Runtime Scheduler Examples\n")

    try:
        example_basic_monitoring()
    except Exception as e:
        print(f"Error in example 1: {e}\n")

    try:
        example_profiling()
    except Exception as e:
        print(f"Error in example 2: {e}\n")

    try:
        example_configuration()
    except Exception as e:
        print(f"Error in example 3: {e}\n")

    try:
        example_data_collection()
    except Exception as e:
        print(f"Error in example 4: {e}\n")

    try:
        example_hooks()
    except Exception as e:
        print(f"Error in example 5: {e}\n")

    try:
        example_complete_workflow()
    except Exception as e:
        print(f"Error in example 6: {e}\n")

    print("All examples completed!")


if __name__ == "__main__":
    main()
