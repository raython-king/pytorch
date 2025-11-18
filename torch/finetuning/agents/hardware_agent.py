"""
Hardware Analysis Agent

Analyzes hardware capabilities for fine-tuning.
"""

import torch
from typing import Dict, Any
from .base_agent import BaseFineTuningAgent, FineTuningDecision


class HardwareAnalysisAgent(BaseFineTuningAgent):
    """Agent that analyzes hardware for fine-tuning"""

    def observe(self, environment: Dict[str, Any]) -> None:
        """Observe hardware state"""
        # Detect GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_memory = 0

        self.state = {
            'has_gpu': torch.cuda.is_available(),
            'gpu_memory_gb': gpu_memory,
            'num_gpus': torch.cuda.device_count(),
            'model_size_gb': environment.get('model_size', 1.0),
        }

    def decide(self) -> FineTuningDecision:
        """Recommend method based on hardware"""
        gpu_memory = self.state.get('gpu_memory_gb', 0)
        model_size = self.state.get('model_size_gb', 1.0)

        if gpu_memory == 0:
            return FineTuningDecision(
                method='lora',
                confidence=0.6,
                config={'r': 4},
                reasoning="No GPU detected, using minimal LoRA"
            )

        memory_ratio = gpu_memory / model_size

        if memory_ratio < 1.5:
            return FineTuningDecision(
                method='qlora',
                confidence=0.9,
                config={'load_in_4bit': True, 'r': 4},
                reasoning=f"Limited memory ({memory_ratio:.1f}x model size), using QLoRA"
            )
        elif memory_ratio < 3.0:
            return FineTuningDecision(
                method='lora',
                confidence=0.85,
                config={'r': 8},
                reasoning=f"Moderate memory ({memory_ratio:.1f}x), using LoRA"
            )
        else:
            return FineTuningDecision(
                method='lora',
                confidence=0.8,
                config={'r': 16},
                reasoning=f"Ample memory ({memory_ratio:.1f}x), using LoRA with higher rank"
            )

    def learn(self, reward: float, next_state: Dict[str, Any]) -> None:
        """Learn from hardware performance"""
        pass
