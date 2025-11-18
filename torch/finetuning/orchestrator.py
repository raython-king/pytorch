"""
Fine-tuning Orchestrator

Coordinates multi-agent fine-tuning method selection and application.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time

from .config import FineTuningConfig, FineTuningMethod
from .agents import (
    MethodSelectorAgent,
    HardwareAnalysisAgent,
    PerformanceMonitoringAgent,
    FineTuningCoordinator,
)
from .lora import inject_lora, save_lora, load_lora, print_lora_stats
from .methods import inject_adapters, inject_prefix_tuning, inject_prompt_tuning, inject_ia3, configure_bitfit


class FineTuningOrchestrator:
    """
    Main orchestrator for multi-agent fine-tuning.

    Coordinates multiple agents to select and apply the best fine-tuning method.
    """

    def __init__(self, config: Optional[FineTuningConfig] = None):
        self.config = config or FineTuningConfig()
        self.config.validate()

        # Initialize agents
        self.method_selector = MethodSelectorAgent(
            "method_selector",
            self.config.to_dict()
        )
        self.hardware_agent = HardwareAnalysisAgent(
            "hardware_agent",
            self.config.to_dict()
        )
        self.performance_agent = PerformanceMonitoringAgent(
            "performance_agent",
            self.config.to_dict()
        )
        self.coordinator = FineTuningCoordinator(
            "coordinator",
            self.config.to_dict()
        )

        # State
        self.model: Optional[nn.Module] = None
        self.applied_method: Optional[str] = None
        self.iteration = 0

        # Integration with memory optimizer
        self.memory_optimizer = None
        if self.config.integrate_with_memory_optimizer:
            try:
                from torch.memory_optimization import MemoryOptimizer
                self.memory_optimizer = MemoryOptimizer(auto_detect=True)
            except ImportError:
                pass

    def prepare_model(
        self,
        model: nn.Module,
        method: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """
        Prepare model for fine-tuning.

        Args:
            model: Model to prepare
            method: Specific method to use (None for automatic selection)
            **kwargs: Additional configuration

        Returns:
            Model with fine-tuning method applied
        """
        self.model = model

        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        # Gather environment
        environment = self._gather_environment(model, model_size)

        # Apply memory optimization first if enabled
        if self.memory_optimizer:
            print("Applying memory optimization...")
            model = self.memory_optimizer.optimize_model(model)

        # Use specified method or auto-select
        if method is None and self.config.auto_select_method:
            # Let agents decide
            decision = self._get_agent_decision(environment)
            method = decision.method
            method_config = decision.config
            print(f"Auto-selected method: {method} (confidence: {decision.confidence:.2f})")
            print(f"Reasoning: {decision.reasoning}")
        else:
            method = method or self.config.method.value
            method_config = kwargs

        # Apply fine-tuning method
        model = self._apply_method(model, method, method_config)

        self.applied_method = method

        # Print statistics
        self._print_stats(model, method)

        return model

    def _gather_environment(self, model: nn.Module, model_size: float) -> Dict[str, Any]:
        """Gather environment information"""
        # GPU memory
        available_memory = 0
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            'model': model,
            'model_size': model_size,
            'available_memory': available_memory,
            'model_type': model.__class__.__name__,
            'prefer_memory_efficiency': self.config.prefer_memory_efficiency,
            'prefer_training_speed': self.config.prefer_training_speed,
            'prefer_accuracy': self.config.prefer_accuracy,
            'max_trainable_params_ratio': self.config.max_trainable_params_ratio,
        }

    def _get_agent_decision(self, environment: Dict[str, Any]):
        """Get coordinated decision from agents"""
        # Agents observe
        self.method_selector.observe(environment)
        self.hardware_agent.observe(environment)

        # Agents decide
        agent_decisions = {
            'method_selector': self.method_selector.decide(),
            'hardware_agent': self.hardware_agent.decide(),
        }

        # Coordinator makes final decision
        self.coordinator.observe({'agent_decisions': agent_decisions})
        return self.coordinator.decide()

    def _apply_method(
        self,
        model: nn.Module,
        method: str,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Apply fine-tuning method to model"""
        print(f"\nApplying {method} fine-tuning...")

        if method == 'lora' or method == FineTuningMethod.LORA.value:
            from .config import LoRAConfig
            lora_config = LoRAConfig(**{k: v for k, v in config.items() if hasattr(LoRAConfig, k)})
            return inject_lora(model, lora_config)

        elif method == 'qlora' or method == FineTuningMethod.QLORA.value:
            # QLoRA requires quantization which needs bitsandbytes
            try:
                import bitsandbytes as bnb
                from .config import QLoRAConfig
                qlora_config = QLoRAConfig(**{k: v for k, v in config.items() if hasattr(QLoRAConfig, k)})
                # Apply quantization then LoRA
                return inject_lora(model, qlora_config)
            except ImportError:
                print("Warning: bitsandbytes not installed, falling back to LoRA")
                from .config import LoRAConfig
                return inject_lora(model, LoRAConfig())

        elif method == 'adapter' or method == FineTuningMethod.ADAPTER.value:
            from .config import AdapterConfig
            adapter_config = AdapterConfig(**{k: v for k, v in config.items() if hasattr(AdapterConfig, k)})
            return inject_adapters(model, adapter_config)

        elif method == 'prefix_tuning' or method == FineTuningMethod.PREFIX_TUNING.value:
            from .config import PrefixTuningConfig
            prefix_config = PrefixTuningConfig(**{k: v for k, v in config.items() if hasattr(PrefixTuningConfig, k)})
            return inject_prefix_tuning(model, prefix_config)

        elif method == 'prompt_tuning' or method == FineTuningMethod.PROMPT_TUNING.value:
            from .config import PromptTuningConfig
            prompt_config = PromptTuningConfig(**{k: v for k, v in config.items() if hasattr(PromptTuningConfig, k)})
            return inject_prompt_tuning(model, prompt_config)

        elif method == 'ia3' or method == FineTuningMethod.IA3.value:
            from .config import IA3Config
            ia3_config = IA3Config(**{k: v for k, v in config.items() if hasattr(IA3Config, k)})
            return inject_ia3(model, ia3_config)

        elif method == 'bitfit' or method == FineTuningMethod.BITFIT.value:
            return configure_bitfit(model)

        else:
            raise ValueError(f"Unknown fine-tuning method: {method}")

    def _print_stats(self, model: nn.Module, method: str):
        """Print fine-tuning statistics"""
        if method in ['lora', 'qlora']:
            print_lora_stats(model)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("\n" + "=" * 80)
            print(f"{method.upper()} Fine-tuning Statistics")
            print("=" * 80)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable ratio: {trainable_params/total_params:.2%}")
            print("=" * 80 + "\n")

    def adapt(self, metrics: Dict[str, float]):
        """Adapt based on training metrics"""
        self.iteration += 1

        if self.iteration % self.config.adaptation_interval != 0:
            return

        # Performance agent observes
        self.performance_agent.observe(metrics)
        decision = self.performance_agent.decide()

        # Calculate reward
        reward = metrics.get('accuracy', 0) - metrics.get('loss', 1.0)

        # Agents learn
        self.method_selector.learn(reward, {})
        self.hardware_agent.learn(reward, {})
        self.performance_agent.learn(reward, {})
        self.coordinator.learn(reward, {})

    def save(self, path: str):
        """Save fine-tuned model or adapters"""
        if self.applied_method in ['lora', 'qlora']:
            save_lora(self.model, path, merge_before_save=self.config.save_merged_model)
        else:
            # Save full model
            torch.save(self.model.state_dict(), path)

    def get_summary(self) -> Dict[str, Any]:
        """Get fine-tuning summary"""
        return {
            'applied_method': self.applied_method,
            'iterations': self.iteration,
            'agent_weights': self.coordinator.agent_weights if hasattr(self.coordinator, 'agent_weights') else {},
        }


class FineTuner:
    """Simplified interface for fine-tuning"""

    def __init__(
        self,
        method: Optional[str] = None,
        auto_detect: bool = True,
        config: Optional[FineTuningConfig] = None,
        **kwargs
    ):
        if config is None:
            config = FineTuningConfig()
            config.auto_select_method = auto_detect

        if method:
            config.method = FineTuningMethod(method)

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.orchestrator = FineTuningOrchestrator(config)

    def prepare_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Prepare model for fine-tuning"""
        return self.orchestrator.prepare_model(model, **kwargs)

    def save(self, path: str):
        """Save fine-tuned weights"""
        self.orchestrator.save(path)

    def summary(self) -> Dict[str, Any]:
        """Get summary"""
        return self.orchestrator.get_summary()
