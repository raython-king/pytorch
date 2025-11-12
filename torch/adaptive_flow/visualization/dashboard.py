"""
Real-time Flow Dashboard

Provides a web-based dashboard for monitoring adaptive flow control in real-time.
Includes flow visualization, topology display, and performance metrics.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from ..flow_monitor import FlowMetricsCollector, LinkUtilizationTracker, PerformanceAnalyzer
from ..policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for dashboard requests"""

    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self._serve_index()
        elif path == '/api/flows':
            self._serve_flows()
        elif path == '/api/links':
            self._serve_links()
        elif path == '/api/performance':
            self._serve_performance()
        elif path == '/api/topology':
            self._serve_topology()
        else:
            self.send_error(404, 'Not Found')

    def _serve_index(self):
        """Serve dashboard HTML"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Adaptive Flow Control Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            color: #4CAF50;
            font-size: 18px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
        }
        .metric-value {
            color: #333;
        }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-critical { color: #f44336; }
        .refresh-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .refresh-button:hover {
            background-color: #45a049;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        .auto-refresh {
            float: right;
            margin-top: -40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Adaptive Flow Control Dashboard</h1>
        <div class="auto-refresh">
            <button class="refresh-button" onclick="refreshAll()">Refresh Now</button>
            <label>
                <input type="checkbox" id="autoRefresh" checked> Auto-refresh (5s)
            </label>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h2>System Overview</h2>
                <div id="overview">Loading...</div>
            </div>

            <div class="card">
                <h2>Performance Metrics</h2>
                <div id="performance">Loading...</div>
            </div>

            <div class="card">
                <h2>Active Flows</h2>
                <div id="flows">Loading...</div>
            </div>

            <div class="card">
                <h2>Link Utilization</h2>
                <div id="links">Loading...</div>
            </div>

            <div class="card" style="grid-column: 1 / -1;">
                <h2>Network Topology</h2>
                <div id="topology">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function formatBps(bps) {
            if (bps === 0) return '0 bps';
            const k = 1000;
            const sizes = ['bps', 'Kbps', 'Mbps', 'Gbps', 'Tbps'];
            const i = Math.floor(Math.log(bps) / Math.log(k));
            return parseFloat((bps / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function formatLatency(seconds) {
            return (seconds * 1000).toFixed(2) + ' ms';
        }

        async function refreshFlows() {
            try {
                const response = await fetch('/api/flows');
                const data = await response.json();

                let html = '<table><tr><th>Flow ID</th><th>Route</th><th>Throughput</th><th>Latency (P95)</th></tr>';
                for (const [flowId, flow] of Object.entries(data)) {
                    html += `<tr>
                        <td>${flowId}</td>
                        <td>${flow.src_device} → ${flow.dst_device}</td>
                        <td>${formatBps(flow.throughput_bps)}</td>
                        <td>${formatLatency(flow.latency_p95)}</td>
                    </tr>`;
                }
                html += '</table>';
                document.getElementById('flows').innerHTML = html;
            } catch (error) {
                document.getElementById('flows').innerHTML = 'Error loading flows';
            }
        }

        async function refreshLinks() {
            try {
                const response = await fetch('/api/links');
                const data = await response.json();

                let html = '<table><tr><th>Link</th><th>Utilization</th><th>Active Flows</th></tr>';
                for (const [linkId, link] of Object.entries(data)) {
                    const utilPct = (link.utilization * 100).toFixed(1);
                    const statusClass = link.utilization > 0.8 ? 'status-critical' :
                                       link.utilization > 0.6 ? 'status-warning' : 'status-good';
                    html += `<tr>
                        <td>${link.src_device} → ${link.dst_device}</td>
                        <td class="${statusClass}">${utilPct}%</td>
                        <td>${link.flow_count}</td>
                    </tr>`;
                }
                html += '</table>';
                document.getElementById('links').innerHTML = html;
            } catch (error) {
                document.getElementById('links').innerHTML = 'Error loading links';
            }
        }

        async function refreshPerformance() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();

                let html = '';
                html += `<div class="metric"><span class="metric-label">Fairness Index:</span><span class="metric-value">${data.fairness_index.toFixed(3)}</span></div>`;
                html += `<div class="metric"><span class="metric-label">Avg Link Utilization:</span><span class="metric-value">${(data.average_link_utilization * 100).toFixed(1)}%</span></div>`;
                html += `<div class="metric"><span class="metric-label">Bottlenecks:</span><span class="metric-value ${data.bottleneck_count > 0 ? 'status-warning' : 'status-good'}">${data.bottleneck_count}</span></div>`;

                if (data.issues && data.issues.length > 0) {
                    html += '<h3 style="margin-top: 15px;">Issues:</h3>';
                    for (const issue of data.issues) {
                        const statusClass = issue.severity === 'high' ? 'status-critical' :
                                          issue.severity === 'medium' ? 'status-warning' : '';
                        html += `<div class="metric ${statusClass}">${issue.message}</div>`;
                    }
                }

                document.getElementById('performance').innerHTML = html;
            } catch (error) {
                document.getElementById('performance').innerHTML = 'Error loading performance data';
            }
        }

        async function refreshOverview() {
            try {
                const [flowsResp, linksResp] = await Promise.all([
                    fetch('/api/flows'),
                    fetch('/api/links')
                ]);
                const flows = await flowsResp.json();
                const links = await linksResp.json();

                let totalBytes = 0;
                let totalTransfers = 0;
                for (const flow of Object.values(flows)) {
                    totalBytes += flow.bytes_sent;
                    totalTransfers += flow.transfers_completed;
                }

                let html = '';
                html += `<div class="metric"><span class="metric-label">Active Flows:</span><span class="metric-value">${Object.keys(flows).length}</span></div>`;
                html += `<div class="metric"><span class="metric-label">Network Links:</span><span class="metric-value">${Object.keys(links).length}</span></div>`;
                html += `<div class="metric"><span class="metric-label">Total Transfers:</span><span class="metric-value">${totalTransfers}</span></div>`;
                html += `<div class="metric"><span class="metric-label">Total Data:</span><span class="metric-value">${formatBytes(totalBytes)}</span></div>`;

                document.getElementById('overview').innerHTML = html;
            } catch (error) {
                document.getElementById('overview').innerHTML = 'Error loading overview';
            }
        }

        async function refreshTopology() {
            try {
                const response = await fetch('/api/topology');
                const data = await response.json();

                let html = '<svg width="100%" height="300" style="border: 1px solid #ddd; border-radius: 4px;">';

                // Simple topology visualization
                const devices = data.devices;
                const deviceCount = devices.length;
                const spacing = 100;

                devices.forEach((device, i) => {
                    const x = 100 + i * spacing;
                    const y = 150;

                    html += `<circle cx="${x}" cy="${y}" r="30" fill="#4CAF50" stroke="#333" stroke-width="2"/>`;
                    html += `<text x="${x}" y="${y + 5}" text-anchor="middle" fill="white" font-size="12">${device}</text>`;
                });

                html += '</svg>';
                document.getElementById('topology').innerHTML = html;
            } catch (error) {
                document.getElementById('topology').innerHTML = 'Error loading topology';
            }
        }

        function refreshAll() {
            refreshFlows();
            refreshLinks();
            refreshPerformance();
            refreshOverview();
            refreshTopology();
        }

        // Auto-refresh
        setInterval(() => {
            if (document.getElementById('autoRefresh').checked) {
                refreshAll();
            }
        }, 5000);

        // Initial load
        refreshAll();
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_flows(self):
        """Serve flow metrics data"""
        if hasattr(self.server, 'flow_collector'):
            flows = self.server.flow_collector.get_all_flows()
            data = {
                flow_id: metrics.to_dict()
                for flow_id, metrics in flows.items()
            }
        else:
            data = {}

        self._send_json(data)

    def _serve_links(self):
        """Serve link metrics data"""
        if hasattr(self.server, 'link_tracker'):
            links = self.server.link_tracker.get_all_links()
            data = {
                link_id: metrics.to_dict()
                for link_id, metrics in links.items()
            }
        else:
            data = {}

        self._send_json(data)

    def _serve_performance(self):
        """Serve performance analysis data"""
        if hasattr(self.server, 'performance_analyzer'):
            data = self.server.performance_analyzer.analyze_performance()
        else:
            data = {}

        self._send_json(data)

    def _serve_topology(self):
        """Serve topology data"""
        if hasattr(self.server, 'link_tracker'):
            links = self.server.link_tracker.get_all_links()
            devices = set()
            for link in links.values():
                devices.add(link.src_device)
                devices.add(link.dst_device)

            data = {
                'devices': sorted(list(devices)),
                'links': [
                    {
                        'src': link.src_device,
                        'dst': link.dst_device,
                        'utilization': link.utilization
                    }
                    for link in links.values()
                ]
            }
        else:
            data = {'devices': [], 'links': []}

        self._send_json(data)

    def _send_json(self, data: dict):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())


class FlowDashboard:
    """
    Real-time dashboard for adaptive flow control

    Provides web interface for monitoring flows, links, and performance.
    """

    def __init__(self, flow_collector: FlowMetricsCollector,
                 link_tracker: LinkUtilizationTracker,
                 performance_analyzer: PerformanceAnalyzer,
                 port: int = 8080):
        """
        Initialize dashboard

        Args:
            flow_collector: Flow metrics collector
            link_tracker: Link utilization tracker
            performance_analyzer: Performance analyzer
            port: HTTP server port
        """
        self.flow_collector = flow_collector
        self.link_tracker = link_tracker
        self.performance_analyzer = performance_analyzer
        self.port = port

        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        logger.info(f"FlowDashboard initialized on port {port}")

    def start(self) -> None:
        """Start the dashboard server"""
        if self.running:
            logger.warning("Dashboard already running")
            return

        self.server = HTTPServer(('localhost', self.port), DashboardHandler)

        # Attach components to server for access by handler
        self.server.flow_collector = self.flow_collector
        self.server.link_tracker = self.link_tracker
        self.server.performance_analyzer = self.performance_analyzer

        self.server_thread = threading.Thread(target=self._serve, daemon=True)
        self.server_thread.start()

        self.running = True
        logger.info(f"Dashboard started at http://localhost:{self.port}")

    def stop(self) -> None:
        """Stop the dashboard server"""
        if not self.running:
            logger.warning("Dashboard not running")
            return

        if self.server:
            self.server.shutdown()
            self.server = None

        if self.server_thread:
            self.server_thread.join(timeout=5.0)
            self.server_thread = None

        self.running = False
        logger.info("Dashboard stopped")

    def _serve(self) -> None:
        """Server thread function"""
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")


# Global dashboard instance
_dashboard_instance: Optional[FlowDashboard] = None


def start_dashboard(flow_collector: FlowMetricsCollector,
                   link_tracker: LinkUtilizationTracker,
                   performance_analyzer: PerformanceAnalyzer,
                   port: int = 8080) -> FlowDashboard:
    """
    Start the global dashboard instance

    Args:
        flow_collector: Flow metrics collector
        link_tracker: Link utilization tracker
        performance_analyzer: Performance analyzer
        port: HTTP server port

    Returns:
        FlowDashboard instance

    Example:
        >>> from torch.adaptive_flow.visualization import start_dashboard
        >>> dashboard = start_dashboard(collector, tracker, analyzer)
        >>> # Dashboard available at http://localhost:8080
    """
    global _dashboard_instance

    if _dashboard_instance is not None and _dashboard_instance.running:
        logger.warning("Dashboard already running")
        return _dashboard_instance

    _dashboard_instance = FlowDashboard(
        flow_collector, link_tracker, performance_analyzer, port
    )
    _dashboard_instance.start()

    return _dashboard_instance


def stop_dashboard() -> None:
    """
    Stop the global dashboard instance

    Example:
        >>> from torch.adaptive_flow.visualization import stop_dashboard
        >>> stop_dashboard()
    """
    global _dashboard_instance

    if _dashboard_instance is not None:
        _dashboard_instance.stop()
        _dashboard_instance = None


__all__ = ['FlowDashboard', 'start_dashboard', 'stop_dashboard']
