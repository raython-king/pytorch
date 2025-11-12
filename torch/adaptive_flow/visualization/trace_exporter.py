"""
Trace Export for Adaptive Flow Control

Exports flow traces in various formats:
- Chrome Trace Format (for chrome://tracing)
- TensorBoard format
- CSV for analysis
- JSON for programmatic access
"""

import time
import json
import csv
import logging
from typing import Dict, List, Optional, Any, TextIO
from pathlib import Path
from dataclasses import dataclass, asdict

from ..flow_monitor import FlowMetricsCollector, LinkUtilizationTracker

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Single trace event"""
    name: str
    cat: str  # category
    ph: str  # phase (B=begin, E=end, X=complete, i=instant)
    ts: float  # timestamp in microseconds
    pid: int = 0  # process id
    tid: int = 0  # thread id
    dur: Optional[float] = None  # duration in microseconds
    args: Dict[str, Any] = None  # event arguments

    def __post_init__(self):
        if self.args is None:
            self.args = {}


class TraceExporter:
    """
    Export flow traces in various formats

    Supports Chrome Trace Format, TensorBoard, CSV, and JSON exports.
    """

    def __init__(self, flow_collector: FlowMetricsCollector,
                 link_tracker: LinkUtilizationTracker):
        """
        Initialize trace exporter

        Args:
            flow_collector: Flow metrics collector
            link_tracker: Link utilization tracker
        """
        self.flow_collector = flow_collector
        self.link_tracker = link_tracker
        self.events: List[TraceEvent] = []
        self.start_time = time.time()

        logger.info("TraceExporter initialized")

    def record_transfer_event(self, flow_id: str, start_time: float,
                              end_time: float, bytes_transferred: int) -> None:
        """
        Record a transfer event

        Args:
            flow_id: Flow identifier
            start_time: Transfer start time (seconds)
            end_time: Transfer end time (seconds)
            bytes_transferred: Number of bytes transferred
        """
        # Convert to microseconds relative to start
        ts = (start_time - self.start_time) * 1e6
        dur = (end_time - start_time) * 1e6

        event = TraceEvent(
            name=f"Transfer",
            cat="flow",
            ph="X",  # Complete event
            ts=ts,
            dur=dur,
            pid=0,
            tid=hash(flow_id) % 1000,  # Use flow_id hash as thread id
            args={
                'flow_id': flow_id,
                'bytes': bytes_transferred,
                'bandwidth_mbps': (bytes_transferred * 8 / 1e6) / (dur / 1e6) if dur > 0 else 0
            }
        )

        self.events.append(event)

    def record_congestion_event(self, link_id: str, timestamp: float,
                                utilization: float) -> None:
        """
        Record a congestion event

        Args:
            link_id: Link identifier
            timestamp: Event timestamp (seconds)
            utilization: Link utilization (0.0 to 1.0)
        """
        ts = (timestamp - self.start_time) * 1e6

        event = TraceEvent(
            name="Congestion" if utilization > 0.8 else "Normal",
            cat="link",
            ph="i",  # Instant event
            ts=ts,
            pid=0,
            tid=hash(link_id) % 1000 + 1000,  # Offset thread id for links
            args={
                'link_id': link_id,
                'utilization': utilization
            }
        )

        self.events.append(event)

    def record_policy_decision(self, flow_id: str, timestamp: float,
                              decision: str, parameters: Dict[str, Any]) -> None:
        """
        Record a policy decision event

        Args:
            flow_id: Flow identifier
            timestamp: Decision timestamp (seconds)
            decision: Decision type
            parameters: Decision parameters
        """
        ts = (timestamp - self.start_time) * 1e6

        event = TraceEvent(
            name=decision,
            cat="policy",
            ph="i",
            ts=ts,
            pid=0,
            tid=hash(flow_id) % 1000,
            args={
                'flow_id': flow_id,
                **parameters
            }
        )

        self.events.append(event)

    def export_chrome_trace(self, output_path: Path) -> None:
        """
        Export trace in Chrome Trace Format

        Args:
            output_path: Output file path

        The generated file can be opened in chrome://tracing
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trace_data = {
            'traceEvents': [asdict(event) for event in self.events],
            'displayTimeUnit': 'ms',
            'metadata': {
                'title': 'Adaptive Flow Control Trace',
                'start_time': self.start_time,
                'flow_count': len(self.flow_collector.get_all_flows()),
                'link_count': len(self.link_tracker.get_all_links())
            }
        }

        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Chrome trace exported to {output_path}")

    def export_tensorboard(self, log_dir: Path) -> None:
        """
        Export trace for TensorBoard

        Args:
            log_dir: TensorBoard log directory

        Note: Requires torch.utils.tensorboard
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.error("TensorBoard not available. Install with: pip install tensorboard")
            return

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(log_dir))

        # Export flow metrics
        flows = self.flow_collector.get_all_flows()
        for flow_id, metrics in flows.items():
            step = int(metrics.duration)

            writer.add_scalar(f'flow/{flow_id}/throughput_mbps',
                            metrics.throughput_bps / 1e6, step)
            writer.add_scalar(f'flow/{flow_id}/latency_mean',
                            metrics.latency_mean * 1000, step)
            writer.add_scalar(f'flow/{flow_id}/loss_rate',
                            metrics.loss_rate, step)

        # Export link utilization
        links = self.link_tracker.get_all_links()
        for link_id, metrics in links.items():
            writer.add_scalar(f'link/{link_id}/utilization',
                            metrics.utilization * 100, 0)

        # Export trace events as timeline
        for event in self.events:
            if event.ph == 'X':  # Complete events
                writer.add_scalar(
                    f'transfer/{event.args.get("flow_id", "unknown")}/bandwidth_mbps',
                    event.args.get('bandwidth_mbps', 0),
                    int(event.ts / 1e6)
                )

        writer.close()
        logger.info(f"TensorBoard logs exported to {log_dir}")

    def export_csv(self, output_path: Path) -> None:
        """
        Export trace as CSV

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'timestamp_us', 'category', 'name', 'phase',
                'thread_id', 'duration_us', 'args'
            ])

            # Write events
            for event in self.events:
                writer.writerow([
                    event.ts,
                    event.cat,
                    event.name,
                    event.ph,
                    event.tid,
                    event.dur if event.dur else '',
                    json.dumps(event.args)
                ])

        logger.info(f"CSV trace exported to {output_path}")

    def export_json(self, output_path: Path) -> None:
        """
        Export complete state as JSON

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all data
        flows = self.flow_collector.get_all_flows()
        links = self.link_tracker.get_all_links()

        data = {
            'metadata': {
                'export_time': time.time(),
                'start_time': self.start_time,
                'duration': time.time() - self.start_time,
            },
            'flows': {
                flow_id: metrics.to_dict()
                for flow_id, metrics in flows.items()
            },
            'links': {
                link_id: metrics.to_dict()
                for link_id, metrics in links.items()
            },
            'events': [asdict(event) for event in self.events],
            'summary': self.flow_collector.get_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON trace exported to {output_path}")

    def export_all(self, output_dir: Path, prefix: str = "trace") -> None:
        """
        Export trace in all formats

        Args:
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())

        self.export_chrome_trace(output_dir / f"{prefix}_{timestamp}.json")
        self.export_csv(output_dir / f"{prefix}_{timestamp}.csv")
        self.export_json(output_dir / f"{prefix}_{timestamp}_full.json")

        try:
            self.export_tensorboard(output_dir / f"{prefix}_{timestamp}_tb")
        except Exception as e:
            logger.warning(f"Failed to export TensorBoard logs: {e}")

        logger.info(f"All traces exported to {output_dir}")

    def clear(self) -> None:
        """Clear recorded events"""
        self.events.clear()
        self.start_time = time.time()
        logger.info("Trace events cleared")


# Convenience functions

def export_chrome_trace(flow_collector: FlowMetricsCollector,
                       link_tracker: LinkUtilizationTracker,
                       output_path: Path) -> None:
    """
    Export Chrome trace format

    Args:
        flow_collector: Flow metrics collector
        link_tracker: Link utilization tracker
        output_path: Output file path

    Example:
        >>> from torch.adaptive_flow.visualization import export_chrome_trace
        >>> export_chrome_trace(collector, tracker, "trace.json")
        >>> # Open chrome://tracing and load trace.json
    """
    exporter = TraceExporter(flow_collector, link_tracker)

    # Generate events from current metrics
    flows = flow_collector.get_all_flows()
    for flow_id, metrics in flows.items():
        if metrics.transfers_completed > 0:
            # Create synthetic transfer events
            avg_latency = metrics.latency_mean
            for i in range(min(metrics.transfers_completed, 100)):  # Limit to 100 events
                start = metrics.start_time + i * avg_latency
                end = start + avg_latency
                bytes_per_transfer = metrics.bytes_sent // metrics.transfers_completed

                exporter.record_transfer_event(
                    flow_id, start, end, bytes_per_transfer
                )

    exporter.export_chrome_trace(output_path)


def export_tensorboard(flow_collector: FlowMetricsCollector,
                      link_tracker: LinkUtilizationTracker,
                      log_dir: Path) -> None:
    """
    Export TensorBoard logs

    Args:
        flow_collector: Flow metrics collector
        link_tracker: Link utilization tracker
        log_dir: TensorBoard log directory

    Example:
        >>> from torch.adaptive_flow.visualization import export_tensorboard
        >>> export_tensorboard(collector, tracker, "runs/adaptive_flow")
        >>> # Run: tensorboard --logdir=runs
    """
    exporter = TraceExporter(flow_collector, link_tracker)
    exporter.export_tensorboard(log_dir)


__all__ = [
    'TraceExporter',
    'TraceEvent',
    'export_chrome_trace',
    'export_tensorboard',
]
