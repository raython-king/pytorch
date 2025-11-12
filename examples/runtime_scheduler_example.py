"""
Runtime Scheduler Example

Demonstrates usage of the device and memory management layer
for runtime scheduling in PyTorch.
"""

import time
import torch
from torch.runtime_scheduler import (
    get_device_manager,
    get_memory_scheduler,
    get_stream_manager,
    get_transfer_optimizer,
    OperationType,
    StreamPriority,
    TransferPriority,
)


def example_device_management():
    """Example: Device management"""
    print("=" * 70)
    print("Example 1: Device Management")
    print("=" * 70)

    dm = get_device_manager()

    # Get device statistics
    print("\nDevice Statistics:")
    stats = dm.get_stats()
    print(f"Total devices: {stats['total_devices']}")
    print(f"CUDA devices: {stats['cuda_devices']}")

    for device_stat in stats['devices']:
        print(f"\nDevice: {device_stat['device']}")
        print(f"  Load: {device_stat['load']:.2%}")
        print(f"  Utilization: {device_stat['utilization']:.2%}")
        print(f"  Memory used: {device_stat['memory_used'] / 1024 / 1024:.2f} MB")
        print(f"  Memory free: {device_stat['memory_free'] / 1024 / 1024:.2f} MB")
        print(f"  Active ops: {device_stat['active_ops']}")
        print(f"  Completed ops: {device_stat['completed_ops']}")

    # Select device for operation
    print("\n\nSelecting device for matrix multiplication...")
    device = dm.select_device(
        op_type="matmul",
        input_sizes=[1024, 1024],
        required_memory=1024 * 1024 * 4,  # 4MB
    )
    print(f"Selected device: {device}")

    # Simulate operation execution
    print("\nSimulating operation execution...")
    op_id = 1
    dm.record_op_start(device, op_id)
    time.sleep(0.01)  # Simulate work
    dm.record_op_end(device, op_id, duration=0.01)
    print("Operation completed!")


def example_memory_management():
    """Example: Memory management"""
    print("\n\n" + "=" * 70)
    print("Example 2: Memory Management")
    print("=" * 70)

    ms = get_memory_scheduler()

    device = torch.device("cpu")

    # Allocate memory
    print("\nAllocating memory blocks...")
    blocks = []
    for i in range(5):
        block = ms.allocate(
            size=1024 * 1024,  # 1MB
            device=device,
            tensor_id=i,
            shape=(256, 1024),
            dtype=torch.float32,
        )
        if block:
            blocks.append(block)
            print(f"  Allocated block {i}: {block.size / 1024 / 1024:.2f} MB")

    # Get memory statistics
    print("\n\nMemory Statistics:")
    stats = ms.get_stats()

    for device_str, pool_stats in stats['pools'].items():
        print(f"\nDevice: {device_str}")
        print(f"  Capacity: {pool_stats['capacity'] / 1024 / 1024:.2f} MB")
        print(f"  Allocated: {pool_stats['allocated'] / 1024 / 1024:.2f} MB")
        print(f"  Free: {pool_stats['free'] / 1024 / 1024:.2f} MB")
        print(f"  Utilization: {pool_stats['utilization']:.2%}")
        print(f"  Total blocks: {pool_stats['total_blocks']}")
        print(f"  Fragmentation: {pool_stats['fragmentation']:.2f}")

    # Record access for prefetching
    print("\n\nRecording tensor accesses...")
    for i in range(3):
        ms.record_access(tensor_id=0, device=device)
        time.sleep(0.001)

    print(f"Prefetch hit rate: {stats['prefetch_hit_rate']:.2%}")

    # Free memory
    print("\n\nFreeing memory blocks...")
    for block in blocks:
        ms.free(block.block_id)
        print(f"  Freed block {block.block_id}")


def example_stream_management():
    """Example: Stream management"""
    print("\n\n" + "=" * 70)
    print("Example 3: Stream Management")
    print("=" * 70)

    sm = get_stream_manager()

    device = torch.device("cpu")

    # Register operations with dependencies
    print("\nRegistering operations...")

    # Operation 1: No dependencies
    op1 = sm.register_operation(
        op_type=OperationType.COMPUTE,
        device=device,
        priority=StreamPriority.NORMAL,
    )
    print(f"  Registered op1: {op1}")

    # Operation 2: Depends on op1
    op2 = sm.register_operation(
        op_type=OperationType.MEMORY,
        device=device,
        dependencies={op1},
        priority=StreamPriority.NORMAL,
    )
    print(f"  Registered op2: {op2} (depends on {op1})")

    # Operation 3: Depends on op2
    op3 = sm.register_operation(
        op_type=OperationType.COMMUNICATION,
        device=device,
        dependencies={op2},
        priority=StreamPriority.HIGH,
    )
    print(f"  Registered op3: {op3} (depends on {op2})")

    # Schedule operations
    print("\n\nScheduling operations...")

    # Schedule op1
    stream1 = sm.schedule_operation(op1)
    if stream1:
        print(f"  op1 scheduled to stream {stream1.stream_id}")

    # Try to schedule op2 (should succeed after op1)
    sm.start_operation(op1)
    sm.complete_operation(op1)

    stream2 = sm.schedule_operation(op2)
    if stream2:
        print(f"  op2 scheduled to stream {stream2.stream_id}")

    # Complete op2 and schedule op3
    sm.start_operation(op2)
    sm.complete_operation(op2)

    stream3 = sm.schedule_operation(op3)
    if stream3:
        print(f"  op3 scheduled to stream {stream3.stream_id}")

    sm.start_operation(op3)
    sm.complete_operation(op3)

    # Get stream statistics
    print("\n\nStream Statistics:")
    stats = sm.get_stats()
    print(f"Total streams: {stats['total_streams']}")
    print(f"Total operations: {stats['total_operations']}")
    print(f"Completed operations: {stats['completed_operations']}")

    for device_str, device_stats in stats['devices'].items():
        print(f"\nDevice: {device_str}")
        print(f"  Streams: {device_stats['streams']}")

        for stream_detail in device_stats['stream_details'][:3]:  # Show first 3
            print(f"\n  Stream {stream_detail['stream_id']}:")
            print(f"    Priority: {stream_detail['priority']}")
            print(f"    Load: {stream_detail['load']:.2%}")
            print(f"    Utilization: {stream_detail['utilization']:.2%}")
            print(f"    Total ops: {stream_detail['total_ops']}")


def example_transfer_optimization():
    """Example: Transfer optimization"""
    print("\n\n" + "=" * 70)
    print("Example 4: Transfer Optimization")
    print("=" * 70)

    to = get_transfer_optimizer()

    src_device = torch.device("cpu")
    dst_device = torch.device("cpu")

    # Submit transfers
    print("\nSubmitting transfers...")
    transfer_ids = []

    for i in range(5):
        transfer_id = to.submit_transfer(
            src_device=src_device,
            dst_device=dst_device,
            size=1024 * 1024 * (i + 1),  # 1MB to 5MB
            priority=TransferPriority.NORMAL,
            can_batch=True,
            tensor_id=i,
        )
        transfer_ids.append(transfer_id)
        print(f"  Submitted transfer {i}: {transfer_id}")

    # Wait for transfers
    print("\n\nWaiting for transfers...")
    for transfer_id in transfer_ids:
        try:
            to.wait_transfer(transfer_id, timeout=0.1)
            print(f"  Transfer {transfer_id} completed")
        except TimeoutError:
            print(f"  Transfer {transfer_id} still pending")

    # Get transfer statistics
    print("\n\nTransfer Statistics:")
    stats = to.get_stats()

    scheduler_stats = stats['scheduler']
    print(f"Total transfers: {scheduler_stats['total_transfers']}")
    print(f"Total bytes: {scheduler_stats['total_bytes'] / 1024 / 1024:.2f} MB")
    print(f"Total time: {scheduler_stats['total_time']:.3f} s")

    if scheduler_stats['avg_bandwidth'] > 0:
        print(f"Avg bandwidth: {scheduler_stats['avg_bandwidth'] / 1e9:.2f} GB/s")

    pinned_stats = stats['pinned_memory']
    print(f"\nPinned Memory:")
    print(f"  Total size: {pinned_stats['total_size'] / 1024 / 1024:.2f} MB")
    print(f"  Pool size: {pinned_stats['pool_size']}")
    print(f"  In use: {pinned_stats['in_use']}")
    print(f"  Free: {pinned_stats['free']}")
    print(f"  Hit rate: {pinned_stats['hit_rate']:.2%}")

    # Check P2P support
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("\n\nP2P Support:")
        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                if i != j:
                    supports = to.supports_p2p(
                        torch.device(f"cuda:{i}"),
                        torch.device(f"cuda:{j}")
                    )
                    print(f"  cuda:{i} -> cuda:{j}: {supports}")


def example_complete_workflow():
    """Example: Complete workflow"""
    print("\n\n" + "=" * 70)
    print("Example 5: Complete Workflow")
    print("=" * 70)

    print("\nInitializing managers...")
    dm = get_device_manager()
    ms = get_memory_scheduler()
    sm = get_stream_manager()
    to = get_transfer_optimizer()

    # Step 1: Select device
    print("\n1. Selecting device...")
    device = dm.select_device(
        op_type="matmul",
        input_sizes=[512, 512],
        required_memory=1024 * 1024,
    )
    print(f"   Selected: {device}")

    # Step 2: Allocate memory
    print("\n2. Allocating memory...")
    block = ms.allocate(
        size=1024 * 1024,
        device=device,
        tensor_id=1,
        shape=(512, 512),
        dtype=torch.float32,
    )
    print(f"   Allocated block: {block.block_id} ({block.size / 1024 / 1024:.2f} MB)")

    # Step 3: Register operation
    print("\n3. Registering operation...")
    op_id = sm.register_operation(
        op_type=OperationType.COMPUTE,
        device=device,
        priority=StreamPriority.NORMAL,
    )
    print(f"   Registered operation: {op_id}")

    # Step 4: Schedule operation
    print("\n4. Scheduling operation...")
    stream_info = sm.schedule_operation(op_id)
    if stream_info:
        print(f"   Scheduled to stream: {stream_info.stream_id}")

    # Step 5: Execute operation
    print("\n5. Executing operation...")
    dm.record_op_start(device, op_id)
    sm.start_operation(op_id)

    # Simulate computation
    time.sleep(0.01)

    sm.complete_operation(op_id)
    dm.record_op_end(device, op_id, duration=0.01)
    print("   Operation completed")

    # Step 6: Record access
    print("\n6. Recording access for prefetching...")
    ms.record_access(tensor_id=1, device=device)

    # Step 7: Free memory
    print("\n7. Freeing memory...")
    ms.free(block.block_id)
    print("   Memory freed")

    # Step 8: Show final statistics
    print("\n8. Final Statistics:")
    print(f"\n   Device Manager:")
    print(f"     Total devices: {dm.get_stats()['total_devices']}")

    print(f"\n   Memory Scheduler:")
    print(f"     Total blocks: {ms.get_stats()['total_blocks']}")

    print(f"\n   Stream Manager:")
    print(f"     Completed ops: {sm.get_stats()['completed_operations']}")

    print(f"\n   Transfer Optimizer:")
    print(f"     Total transfers: {to.get_stats()['total_transfers']}")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("Runtime Scheduler Examples")
    print("=" * 70)

    try:
        example_device_management()
        example_memory_management()
        example_stream_management()
        example_transfer_optimization()
        example_complete_workflow()

        print("\n\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
