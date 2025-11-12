"""
ML Scheduler Orchestrator

Main entry point for ML-based scheduling decisions. Coordinates feature extraction,
model inference, and integration with the existing scheduler.
"""

from typing import Optional, List, Tuple, Dict, Any
import torch
import logging
from dataclasses import dataclass

from ..scheduler import BaseSchedulerNode
from .features.node_features import NodeFeatureExtractor
from .features.edge_features import EdgeFeatureExtractor
from .features.graph_features import GraphFeatureExtractor
from .inference.predictor import EnsemblePredictor
from .inference.confidence import ConfidenceScorer
from .inference.fallback import FallbackDecider
from .config import MLSchedulerConfig

log = logging.getLogger(__name__)


@dataclass
class FusionPlan:
    """Represents a fusion decision plan for a set of nodes."""
    fusions: List[Tuple[int, int]]  # Pairs of node indices to fuse
    confidence_scores: List[float]  # Confidence for each fusion
    schedule_order: Optional[List[int]] = None  # Recommended execution order
    metadata: Optional[Dict[str, Any]] = None  # Additional info


class MLSchedulerOrchestrator:
    """
    Main orchestrator for ML-based scheduling.
    
    Responsibilities:
    1. Extract features from IR graphs
    2. Coordinate model inference
    3. Generate fusion plans
    4. Manage fallback to heuristics
    5. Collect feedback for online learning
    
    Example Usage:
        orchestrator = MLSchedulerOrchestrator()
        fusion_plan = orchestrator.predict_fusion_plan(nodes, device)
        if fusion_plan.is_confident():
            apply_fusion_plan(nodes, fusion_plan)
        else:
            use_heuristic_fallback()
    """
    
    def __init__(self, config: Optional[MLSchedulerConfig] = None):
        """
        Initialize ML scheduler orchestrator.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or MLSchedulerConfig()
        
        # Feature extractors
        self.node_feature_extractor = NodeFeatureExtractor()
        self.edge_feature_extractor = EdgeFeatureExtractor()
        self.graph_feature_extractor = GraphFeatureExtractor()
        
        # Model ensemble
        self.predictor = self._load_predictor()
        
        # Confidence and fallback
        self.confidence_scorer = ConfidenceScorer(
            threshold=self.config.confidence_threshold
        )
        self.fallback_decider = FallbackDecider()
        
        # Feedback collection
        self.feedback_buffer = []
        self.prediction_cache = {}
        
        log.info(f"MLSchedulerOrchestrator initialized with config: {self.config}")
    
    def _load_predictor(self) -> EnsemblePredictor:
        """Load trained model ensemble."""
        if self.config.model_path is None:
            log.warning("No model path specified, using random initialization")
            return EnsemblePredictor(pretrained=False)
        
        try:
            predictor = EnsemblePredictor.load(self.config.model_path)
            log.info(f"Loaded model from {self.config.model_path}")
            return predictor
        except Exception as e:
            log.error(f"Failed to load model: {e}, using random initialization")
            return EnsemblePredictor(pretrained=False)
    
    def predict_fusion_plan(
        self,
        nodes: List[BaseSchedulerNode],
        device: torch.device
    ) -> FusionPlan:
        """
        Generate fusion plan for a list of nodes.
        
        Args:
            nodes: List of scheduler nodes to fuse
            device: Target device
            
        Returns:
            FusionPlan with fusion decisions and confidence scores
        """
        # Check cache first
        cache_key = self._get_cache_key(nodes)
        if self.config.cache_predictions and cache_key in self.prediction_cache:
            log.debug(f"Cache hit for {len(nodes)} nodes")
            return self.prediction_cache[cache_key]
        
        try:
            # Extract features
            graph_data = self._extract_graph_features(nodes, device)
            
            # Model inference
            predictions = self.predictor.predict(graph_data)
            
            # Generate fusion plan
            fusion_plan = self._generate_fusion_plan(nodes, predictions)
            
            # Score confidence
            fusion_plan = self.confidence_scorer.score_plan(fusion_plan, graph_data)
            
            # Cache result
            if self.config.cache_predictions:
                self.prediction_cache[cache_key] = fusion_plan
            
            log.debug(
                f"Generated fusion plan for {len(nodes)} nodes: "
                f"{len(fusion_plan.fusions)} fusions with avg confidence "
                f"{sum(fusion_plan.confidence_scores)/len(fusion_plan.confidence_scores):.3f}"
            )
            
            return fusion_plan
            
        except Exception as e:
            log.error(f"Error in predict_fusion_plan: {e}")
            # Return empty plan to trigger fallback
            return FusionPlan(fusions=[], confidence_scores=[], metadata={'error': str(e)})
    
    def predict_fusion_candidates(
        self,
        nodes: List[BaseSchedulerNode],
        return_confidence: bool = True
    ) -> List[Tuple[BaseSchedulerNode, BaseSchedulerNode, float]]:
        """
        Predict fusion candidates with confidence scores.
        
        Args:
            nodes: List of scheduler nodes
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of (node1, node2, confidence) tuples
        """
        fusion_plan = self.predict_fusion_plan(nodes, nodes[0].get_device())
        
        candidates = []
        for (i, j), confidence in zip(fusion_plan.fusions, fusion_plan.confidence_scores):
            candidates.append((nodes[i], nodes[j], confidence))
        
        return candidates
    
    def predict_fusion_score(self, pairwise_features: torch.Tensor) -> float:
        """
        Predict fusion score for a pair of nodes.
        
        Args:
            pairwise_features: Feature tensor for node pair
            
        Returns:
            Fusion score in [0, 1]
        """
        score = self.predictor.predict_pairwise(pairwise_features)
        return float(score)
    
    def extract_pairwise_features(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode
    ) -> torch.Tensor:
        """Extract features for a pair of nodes."""
        # Node features
        node1_features = self.node_feature_extractor.extract_features(node1)
        node2_features = self.node_feature_extractor.extract_features(node2)
        
        # Edge features (if dependency exists)
        edge_features = self.edge_feature_extractor.extract_pairwise_features(
            node1, node2
        )
        
        # Concatenate
        pairwise_features = torch.cat([node1_features, node2_features, edge_features])
        
        return pairwise_features
    
    def _extract_graph_features(
        self,
        nodes: List[BaseSchedulerNode],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive graph features.
        
        Returns:
            Dictionary with:
            - 'x': Node features [num_nodes, node_dim]
            - 'edge_index': Edge connectivity [2, num_edges]
            - 'edge_attr': Edge features [num_edges, edge_dim]
            - 'global_features': Graph-level features [graph_dim]
        """
        num_nodes = len(nodes)
        
        # Extract node features
        node_features = []
        for node in nodes:
            features = self.node_feature_extractor.extract_features(node)
            node_features.append(features)
        
        node_features = torch.stack(node_features)  # [num_nodes, node_dim]
        
        # Extract edge features and connectivity
        edge_index = []
        edge_features = []
        
        for i, src_node in enumerate(nodes):
            for j, dst_node in enumerate(nodes):
                if i == j:
                    continue
                
                # Check if edge exists (dependency)
                if self._has_dependency(src_node, dst_node):
                    edge_index.append([i, j])
                    
                    # Extract edge features
                    edge_feat = self.edge_feature_extractor.extract_features(
                        src_node, dst_node, None  # dependency object
                    )
                    edge_features.append(edge_feat)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, num_edges]
            edge_features = torch.stack(edge_features)  # [num_edges, edge_dim]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((0, self.edge_feature_extractor.feature_dim))
        
        # Extract global features
        global_features = self.graph_feature_extractor.extract_features(nodes)
        
        return {
            'x': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features,
            'global_features': global_features,
            'num_nodes': num_nodes,
            'device': device,
        }
    
    def _generate_fusion_plan(
        self,
        nodes: List[BaseSchedulerNode],
        predictions: Dict[str, torch.Tensor]
    ) -> FusionPlan:
        """
        Generate fusion plan from model predictions.
        
        Args:
            nodes: List of nodes
            predictions: Model predictions dict with 'fusion_matrix' [num_nodes, num_nodes]
            
        Returns:
            FusionPlan
        """
        fusion_matrix = predictions['fusion_matrix']  # [num_nodes, num_nodes]
        num_nodes = len(nodes)
        
        # Extract high-confidence fusions
        fusions = []
        confidence_scores = []
        
        # Greedy fusion selection
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                score = fusion_matrix[i, j].item()
                
                if score > 0.5:  # Threshold for fusion
                    fusions.append((i, j))
                    confidence_scores.append(score)
        
        # Sort by confidence
        sorted_indices = sorted(
            range(len(fusions)),
            key=lambda idx: confidence_scores[idx],
            reverse=True
        )
        
        fusions = [fusions[idx] for idx in sorted_indices]
        confidence_scores = [confidence_scores[idx] for idx in sorted_indices]
        
        # Schedule order (if available)
        schedule_order = None
        if 'schedule_order' in predictions:
            schedule_order = predictions['schedule_order'].tolist()
        
        return FusionPlan(
            fusions=fusions,
            confidence_scores=confidence_scores,
            schedule_order=schedule_order,
            metadata={
                'num_nodes': num_nodes,
                'num_candidates': len(fusions),
            }
        )
    
    def _has_dependency(
        self,
        src_node: BaseSchedulerNode,
        dst_node: BaseSchedulerNode
    ) -> bool:
        """Check if dst_node depends on src_node."""
        # Check if any output of src_node is read by dst_node
        src_outputs = {buf.get_name() for buf in src_node.get_outputs()}
        dst_inputs = {dep.name for dep in dst_node.read_writes.reads}
        
        return len(src_outputs & dst_inputs) > 0
    
    def _get_cache_key(self, nodes: List[BaseSchedulerNode]) -> str:
        """Generate cache key for a graph."""
        node_types = tuple(type(n).__name__ for n in nodes)
        num_nodes = len(nodes)
        
        # Simple hash based on structure
        key = f"{num_nodes}_{hash(node_types)}"
        return key
    
    def record_prediction(
        self,
        nodes: List[BaseSchedulerNode],
        fusion_plan: FusionPlan,
        result_nodes: List[BaseSchedulerNode]
    ):
        """
        Record prediction for feedback collection.
        
        Args:
            nodes: Original nodes
            fusion_plan: Predicted fusion plan
            result_nodes: Nodes after applying fusion
        """
        if not self.config.collect_feedback:
            return
        
        feedback = {
            'num_nodes_before': len(nodes),
            'num_nodes_after': len(result_nodes),
            'num_fusions': len(fusion_plan.fusions),
            'avg_confidence': sum(fusion_plan.confidence_scores) / len(fusion_plan.confidence_scores) if fusion_plan.confidence_scores else 0,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
        }
        
        self.feedback_buffer.append(feedback)
        
        # Periodic logging
        if len(self.feedback_buffer) % 100 == 0:
            log.info(f"Collected {len(self.feedback_buffer)} feedback samples")
    
    def should_use_ml(self, num_nodes: int) -> bool:
        """
        Decide whether to use ML scheduler based on graph characteristics.
        
        Args:
            num_nodes: Number of nodes in graph
            
        Returns:
            True if ML scheduler should be used
        """
        # Simple heuristic: only use ML for graphs with enough nodes
        if num_nodes < self.config.min_nodes_for_ml:
            return False
        
        if num_nodes > self.config.max_nodes_for_ml:
            return False
        
        return True
