#!/usr/bin/env python3
"""
Complete End-to-End Example: Train and Deploy ML Scheduler

This script demonstrates the complete workflow:
1. Collect training data from heuristic scheduler
2. Train GNN model
3. Validate in shadow mode
4. Deploy with fallback

Usage:
    # Step 1: Collect data
    python train_and_deploy.py --mode collect --output ./training_data

    # Step 2: Train model
    python train_and_deploy.py --mode train --data ./training_data --output ./checkpoints

    # Step 3: Validate
    python train_and_deploy.py --mode validate --model ./checkpoints/best_model.pt

    # Step 4: Deploy
    python train_and_deploy.py --mode deploy --model ./checkpoints/best_model.pt
"""

import argparse
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def collect_training_data(output_dir: Path, num_samples: int = 1000):
    """
    Collect training data from heuristic scheduler.

    This function would normally hook into the actual PyTorch inductor
    scheduler and collect real compilation graphs. For this example,
    we generate synthetic data.

    Args:
        output_dir: Directory to save training data
        num_samples: Number of graphs to collect
    """
    log.info(f"Collecting {num_samples} training samples...")

    output_dir.mkdir(parents=True, exist_ok=True)

    from torch._inductor.ml_scheduler.training.dataset import save_graph_to_file

    for i in range(num_samples):
        # Generate synthetic graph
        num_nodes = torch.randint(5, 50, (1,)).item()
        num_edges = torch.randint(num_nodes, num_nodes * 3, (1,)).item()

        graph_data = {
            'x': torch.randn(num_nodes, 64),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_attr': torch.randn(num_edges, 32),
            'num_nodes': num_nodes,
            # Ground truth fusion decisions (random for demo)
            'y_fusion': torch.randint(0, 2, (num_nodes, num_nodes)).float(),
            # Ground truth schedule order
            'y_schedule': torch.randperm(num_nodes).float(),
            # Performance metric (compilation time in ms)
            'y_performance': torch.rand(1).item() * 100,
        }

        # Save to file
        output_path = output_dir / f"graph_{i:05d}.pkl"
        save_graph_to_file(graph_data, output_path, format='pkl')

        if (i + 1) % 100 == 0:
            log.info(f"Collected {i + 1}/{num_samples} samples")

    log.info(f"Data collection complete! Saved to {output_dir}")


def train_model(
    data_dir: Path,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 32,
):
    """
    Train ML scheduler model.

    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
    """
    log.info("Starting model training...")

    from torch._inductor.ml_scheduler.training.trainer import MLSchedulerTrainer, TrainingConfig
    from torch._inductor.ml_scheduler.training.dataset import create_dataset_split

    # Create dataset split
    train_dataset, val_dataset, test_dataset = create_dataset_split(
        data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Training configuration
    config = TrainingConfig(
        mode='supervised',
        model_type='fusion_gnn',
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs'),
        save_every=5,
        patience=10,
        use_amp=True,
    )

    # Create trainer
    trainer = MLSchedulerTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Train model
    log.info("Training...")
    trainer.train()

    # Export best model
    best_model_path = output_dir / 'best_model.pt'
    trainer.load_checkpoint(str(output_dir / 'checkpoints' / 'best_model.pt'))
    trainer.export_model(str(best_model_path))

    log.info(f"Training complete! Best model saved to {best_model_path}")


def validate_model(model_path: Path, num_test_cases: int = 100):
    """
    Validate model in shadow mode.

    Args:
        model_path: Path to trained model
        num_test_cases: Number of test cases to run
    """
    log.info(f"Validating model: {model_path}")

    from torch._inductor.ml_scheduler.config import MLSchedulerConfig
    from torch._inductor.ml_scheduler.integration.scheduler_hook import (
        enable_ml_scheduler,
        MLSchedulerMode,
        disable_ml_scheduler,
    )

    # Enable ML scheduler in shadow mode
    config = MLSchedulerConfig(
        model_path=str(model_path),
        confidence_threshold=0.75,
    )

    enable_ml_scheduler(mode=MLSchedulerMode.SHADOW, config=config)

    try:
        # Run test compilations
        log.info(f"Running {num_test_cases} test compilations in shadow mode...")

        for i in range(num_test_cases):
            # Create a simple test model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10),
            )

            x = torch.randn(5, 10)

            # Run inference (would trigger compilation in real scenario)
            with torch.no_grad():
                output = model(x)

            if (i + 1) % 10 == 0:
                log.info(f"Completed {i + 1}/{num_test_cases} test cases")

        log.info("Validation complete!")

        # In real scenario, you would analyze shadow logs here
        log.info("Check shadow logs for ML vs heuristic comparisons")

    finally:
        disable_ml_scheduler()


def deploy_model(model_path: Path, mode: str = 'hybrid'):
    """
    Deploy model for production use.

    Args:
        model_path: Path to trained model
        mode: Deployment mode ('shadow', 'hybrid', 'full')
    """
    log.info(f"Deploying model in {mode} mode: {model_path}")

    from torch._inductor.ml_scheduler.config import MLSchedulerConfig
    from torch._inductor.ml_scheduler.integration.scheduler_hook import (
        enable_ml_scheduler,
        MLSchedulerMode,
    )

    # Configure ML scheduler
    config = MLSchedulerConfig(
        model_path=str(model_path),
        confidence_threshold=0.75,
        fallback_on_error=True,
        validate_fusion_plan=True,
        max_inference_time_ms=50.0,
    )

    # Map mode string to enum
    mode_map = {
        'shadow': MLSchedulerMode.SHADOW,
        'hybrid': MLSchedulerMode.HYBRID,
        'full': MLSchedulerMode.FULL,
    }

    enable_ml_scheduler(mode=mode_map[mode], config=config)

    log.info(f"ML Scheduler deployed successfully in {mode} mode!")
    log.info("The scheduler will now be used for all torch.compile() calls")

    # Example usage
    log.info("\nExample usage:")
    log.info("  import torch")
    log.info("  model = torch.nn.Linear(10, 10)")
    log.info("  compiled_model = torch.compile(model)")
    log.info("  # ML scheduler will optimize compilation")


def benchmark_model(model_path: Path, num_iterations: int = 100):
    """
    Benchmark ML scheduler performance.

    Args:
        model_path: Path to trained model
        num_iterations: Number of benchmark iterations
    """
    log.info(f"Benchmarking model: {model_path}")

    from torch._inductor.ml_scheduler.inference.predictor import MLSchedulerPredictor
    import time

    # Load predictor
    predictor = MLSchedulerPredictor.load(str(model_path))

    # Warmup
    log.info("Warming up...")
    predictor.warmup(num_iterations=10, graph_size=50)

    # Benchmark
    log.info(f"Running {num_iterations} benchmark iterations...")

    for i in range(num_iterations):
        # Create dummy graph
        num_nodes = torch.randint(10, 100, (1,)).item()
        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(num_nodes * 2, 32)

        # Predict
        result = predictor.predict(x, edge_index, edge_attr, use_cache=True)

        if (i + 1) % 10 == 0:
            log.info(f"Completed {i + 1}/{num_iterations} iterations")

    # Get statistics
    stats = predictor.get_performance_stats()

    log.info("\n" + "="*50)
    log.info("Benchmark Results:")
    log.info("="*50)
    log.info(f"Total predictions: {stats['num_predictions']}")
    log.info(f"Mean time: {stats['mean_time_ms']:.2f}ms")
    log.info(f"Median time: {stats['median_time_ms']:.2f}ms")
    log.info(f"P95 time: {stats['p95_time_ms']:.2f}ms")
    log.info(f"P99 time: {stats['p99_time_ms']:.2f}ms")

    if 'cache' in stats:
        log.info(f"\nCache statistics:")
        log.info(f"  Hit rate: {stats['cache']['hit_rate']:.2%}")
        log.info(f"  Hits: {stats['cache']['hits']}")
        log.info(f"  Misses: {stats['cache']['misses']}")

    log.info("="*50)


def analyze_results(checkpoint_dir: Path):
    """
    Analyze training results and generate report.

    Args:
        checkpoint_dir: Directory containing checkpoints and logs
    """
    log.info(f"Analyzing results from: {checkpoint_dir}")

    # Load tensorboard logs
    log_dir = checkpoint_dir / 'logs'

    if not log_dir.exists():
        log.warning(f"Log directory not found: {log_dir}")
        return

    log.info(f"TensorBoard logs available at: {log_dir}")
    log.info("Run: tensorboard --logdir={log_dir}")

    # Check for checkpoints
    checkpoint_files = list((checkpoint_dir / 'checkpoints').glob('*.pt'))

    log.info(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for cp in sorted(checkpoint_files)[-5:]:
        log.info(f"  - {cp.name}")

    # Generate summary report
    report = {
        'checkpoint_dir': str(checkpoint_dir),
        'num_checkpoints': len(checkpoint_files),
        'log_dir': str(log_dir),
    }

    report_path = checkpoint_dir / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    log.info(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='ML Scheduler Training and Deployment Pipeline'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['collect', 'train', 'validate', 'deploy', 'benchmark', 'analyze'],
        help='Operation mode'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data directory'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to collect (collect mode)'
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Number of training epochs (train mode)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (train mode)'
    )

    parser.add_argument(
        '--deploy-mode',
        type=str,
        default='hybrid',
        choices=['shadow', 'hybrid', 'full'],
        help='Deployment mode (deploy mode)'
    )

    parser.add_argument(
        '--num-iterations',
        type=int,
        default=100,
        help='Number of iterations (benchmark mode)'
    )

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == 'collect':
        if not args.output:
            parser.error("--output required for collect mode")

        output_dir = Path(args.output)
        collect_training_data(output_dir, num_samples=args.num_samples)

    elif args.mode == 'train':
        if not args.data or not args.output:
            parser.error("--data and --output required for train mode")

        data_dir = Path(args.data)
        output_dir = Path(args.output)

        train_model(
            data_dir,
            output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )

    elif args.mode == 'validate':
        if not args.model:
            parser.error("--model required for validate mode")

        model_path = Path(args.model)
        validate_model(model_path)

    elif args.mode == 'deploy':
        if not args.model:
            parser.error("--model required for deploy mode")

        model_path = Path(args.model)
        deploy_model(model_path, mode=args.deploy_mode)

    elif args.mode == 'benchmark':
        if not args.model:
            parser.error("--model required for benchmark mode")

        model_path = Path(args.model)
        benchmark_model(model_path, num_iterations=args.num_iterations)

    elif args.mode == 'analyze':
        if not args.output:
            parser.error("--output required for analyze mode")

        checkpoint_dir = Path(args.output)
        analyze_results(checkpoint_dir)


if __name__ == '__main__':
    main()
