"""
Configuration for Multi-Agent Fine-tuning System
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum


class FineTuningMethod(Enum):
    """Available fine-tuning methods"""
    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    IA3 = "ia3"
    BITFIT = "bitfit"
    FULL = "full"


class QuantizationType(Enum):
    """Quantization types for QLoRA"""
    INT4 = "int4"
    INT8 = "int8"
    NF4 = "nf4"  # Normal Float 4-bit
    FP4 = "fp4"  # Float Point 4-bit


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    r: int = 8  # Rank of the low-rank matrices
    alpha: int = 16  # Scaling factor
    dropout: float = 0.1  # Dropout probability

    # Target modules to apply LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Module types to apply LoRA
    target_module_types: Set[str] = field(default_factory=lambda: {
        "Linear", "Conv2d", "Embedding"
    })

    # Merge LoRA weights during inference
    merge_weights: bool = False

    # Fan-in-fan-out mode (for some models like GPT-2)
    fan_in_fan_out: bool = False

    # Bias handling
    bias: str = "none"  # none, all, lora_only

    # Enable LoRA for specific layers
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA)"""
    # Quantization settings
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"  # bfloat16, float16, float32
    bnb_4bit_quant_type: str = "nf4"  # nf4, fp4
    bnb_4bit_use_double_quant: bool = True


@dataclass
class AdapterConfig:
    """Configuration for Adapter-based fine-tuning"""
    bottleneck_size: int = 64  # Size of adapter bottleneck
    non_linearity: str = "gelu"  # Activation function
    adapter_dropout: float = 0.1
    adapter_layernorm_option: str = "in"  # in, out, none

    # Adapter placement
    adapter_residual_before_ln: bool = False

    # Target layers
    target_modules: List[str] = field(default_factory=lambda: [
        "attention", "feed_forward"
    ])


@dataclass
class PrefixTuningConfig:
    """Configuration for Prefix Tuning"""
    num_virtual_tokens: int = 20  # Number of virtual tokens
    prefix_projection: bool = True  # Use MLP for prefix projection
    projection_hidden_size: int = 512
    encoder_hidden_size: int = 768

    # Prefix dropout
    prefix_dropout: float = 0.1


@dataclass
class PromptTuningConfig:
    """Configuration for Prompt Tuning"""
    num_virtual_tokens: int = 8
    prompt_tuning_init: str = "random"  # random, text
    prompt_tuning_init_text: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None


@dataclass
class IA3Config:
    """Configuration for IA3 fine-tuning"""
    target_modules: List[str] = field(default_factory=lambda: [
        "k_proj", "v_proj", "down_proj"
    ])
    feedforward_modules: List[str] = field(default_factory=lambda: [
        "down_proj"
    ])

    # Initialization
    init_ia3_weights: bool = True


@dataclass
class FineTuningConfig:
    """Main configuration for fine-tuning system"""

    # Method selection
    method: FineTuningMethod = FineTuningMethod.LORA
    auto_select_method: bool = True

    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # QLoRA configuration
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)

    # Adapter configuration
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    # Prefix tuning configuration
    prefix_tuning: PrefixTuningConfig = field(default_factory=PrefixTuningConfig)

    # Prompt tuning configuration
    prompt_tuning: PromptTuningConfig = field(default_factory=PromptTuningConfig)

    # IA3 configuration
    ia3: IA3Config = field(default_factory=IA3Config)

    # General settings
    inference_mode: bool = False

    # Multi-agent settings
    use_agent_selection: bool = True
    agent_selection_strategy: str = "performance"  # performance, memory, speed

    # Performance preferences (0-1 scale)
    prefer_memory_efficiency: float = 0.5
    prefer_training_speed: float = 0.3
    prefer_accuracy: float = 0.7

    # Hardware constraints
    min_gpu_memory_gb: Optional[float] = None
    max_trainable_params_ratio: float = 0.1  # Max 10% trainable params

    # Integration with memory optimization
    integrate_with_memory_optimizer: bool = True

    # Monitoring and adaptation
    enable_monitoring: bool = True
    adaptation_interval: int = 100

    # Saving and loading
    save_merged_model: bool = False
    save_adapter_only: bool = True

    def validate(self) -> None:
        """Validate configuration"""
        assert 0.0 <= self.prefer_memory_efficiency <= 1.0
        assert 0.0 <= self.prefer_training_speed <= 1.0
        assert 0.0 <= self.prefer_accuracy <= 1.0
        assert 0.0 < self.max_trainable_params_ratio <= 1.0

        # Validate method-specific configs
        if self.method == FineTuningMethod.LORA:
            assert self.lora.r > 0
            assert self.lora.alpha > 0

        elif self.method == FineTuningMethod.QLORA:
            assert self.qlora.r > 0
            assert self.qlora.load_in_4bit or self.qlora.load_in_8bit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FineTuningConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    @classmethod
    def for_lora(
        cls,
        r: int = 8,
        alpha: int = 16,
        target_modules: Optional[List[str]] = None
    ) -> 'FineTuningConfig':
        """Create LoRA-specific configuration"""
        config = cls()
        config.method = FineTuningMethod.LORA
        config.lora.r = r
        config.lora.alpha = alpha
        if target_modules:
            config.lora.target_modules = target_modules
        return config

    @classmethod
    def for_qlora(
        cls,
        r: int = 8,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16"
    ) -> 'FineTuningConfig':
        """Create QLoRA-specific configuration"""
        config = cls()
        config.method = FineTuningMethod.QLORA
        config.qlora.r = r
        config.qlora.load_in_4bit = load_in_4bit
        config.qlora.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        return config

    @classmethod
    def for_hardware(
        cls,
        gpu_memory_gb: float,
        model_size_gb: float
    ) -> 'FineTuningConfig':
        """Create hardware-optimized configuration"""
        config = cls()

        # Select method based on available memory
        memory_ratio = gpu_memory_gb / model_size_gb

        if memory_ratio < 1.5:
            # Very limited memory - use QLoRA with 4-bit
            config.method = FineTuningMethod.QLORA
            config.qlora.load_in_4bit = True
            config.qlora.r = 4  # Lower rank

        elif memory_ratio < 3.0:
            # Limited memory - use LoRA
            config.method = FineTuningMethod.LORA
            config.lora.r = 8

        elif memory_ratio < 5.0:
            # Moderate memory - use LoRA with higher rank
            config.method = FineTuningMethod.LORA
            config.lora.r = 16

        else:
            # Ample memory - can use adapters or full fine-tuning
            config.method = FineTuningMethod.ADAPTER
            config.adapter.bottleneck_size = 128

        config.validate()
        return config
