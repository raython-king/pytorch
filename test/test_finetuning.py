"""
Tests for Multi-Agent Fine-tuning System
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from torch.finetuning import FineTuner, FineTuningConfig
from torch.finetuning.lora import inject_lora, LinearLoRA, get_lora_stats
from torch.finetuning.config import LoRAConfig


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestLoRALayers(TestCase):
    """Test LoRA layer implementations"""

    def test_linear_lora(self):
        """Test LinearLoRA layer"""
        layer = LinearLoRA(100, 200, r=8, lora_alpha=16)

        x = torch.randn(4, 100)
        y = layer(x)

        self.assertEqual(y.shape, (4, 200))

    def test_lora_merge(self):
        """Test merging LoRA weights"""
        layer = LinearLoRA(100, 200, r=8)

        # Merge weights
        layer.merge()
        self.assertTrue(layer.merged)

        # Unmerge
        layer.unmerge()
        self.assertFalse(layer.merged)

    def test_lora_injection(self):
        """Test injecting LoRA into model"""
        model = SimpleModel()
        config = LoRAConfig(r=4, alpha=8)

        model = inject_lora(model, config)

        # Check that LoRA was injected
        has_lora = any('lora_' in n for n, _ in model.named_parameters())
        self.assertTrue(has_lora)

    def test_lora_stats(self):
        """Test LoRA statistics"""
        model = SimpleModel()
        model = inject_lora(model, LoRAConfig(r=4))

        stats = get_lora_stats(model)

        self.assertGreater(stats['lora_parameters'], 0)
        self.assertGreater(stats['trainable_ratio'], 0)
        self.assertLess(stats['trainable_ratio'], 1.0)


class TestFineTuningConfig(TestCase):
    """Test configuration"""

    def test_config_validation(self):
        """Test config validation"""
        config = FineTuningConfig()
        config.validate()  # Should not raise

    def test_lora_config_creation(self):
        """Test creating LoRA config"""
        config = FineTuningConfig.for_lora(r=16, alpha=32)

        self.assertEqual(config.lora.r, 16)
        self.assertEqual(config.lora.alpha, 32)

    def test_hardware_config(self):
        """Test hardware-specific config"""
        config = FineTuningConfig.for_hardware(
            gpu_memory_gb=8.0,
            model_size_gb=4.0
        )

        self.assertIsNotNone(config.method)


class TestFineTuner(TestCase):
    """Test high-level FineTuner interface"""

    def test_finetuner_creation(self):
        """Test creating fine-tuner"""
        finetuner = FineTuner(method='lora', auto_detect=False)
        self.assertIsNotNone(finetuner)

    def test_prepare_model_lora(self):
        """Test preparing model with LoRA"""
        finetuner = FineTuner(method='lora', auto_detect=False)
        model = SimpleModel()

        model = finetuner.prepare_model(model, r=8)

        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        self.assertGreater(trainable, 0)
        self.assertLess(trainable / total, 0.5)  # LoRA should have <50% trainable

    def test_prepare_model_auto(self):
        """Test automatic method selection"""
        finetuner = FineTuner(auto_detect=True)
        model = SimpleModel()

        model = finetuner.prepare_model(model)

        # Should have selected some method
        summary = finetuner.summary()
        self.assertIsNotNone(summary['applied_method'])

    def test_forward_pass_after_lora(self):
        """Test forward pass after LoRA injection"""
        finetuner = FineTuner(method='lora')
        model = SimpleModel()
        model = finetuner.prepare_model(model, r=4)

        x = torch.randn(2, 100)
        y = model(x)

        self.assertEqual(y.shape, (2, 10))


class TestAgents(TestCase):
    """Test fine-tuning agents"""

    def test_method_selector(self):
        """Test method selector agent"""
        from torch.finetuning.agents import MethodSelectorAgent

        agent = MethodSelectorAgent("test", {})

        environment = {
            'model_size': 1.0,
            'available_memory': 16.0,
            'prefer_memory_efficiency': 0.8,
        }
        agent.observe(environment)

        decision = agent.decide()
        self.assertIsNotNone(decision.method)
        self.assertGreater(decision.confidence, 0)

    def test_hardware_agent(self):
        """Test hardware analysis agent"""
        from torch.finetuning.agents import HardwareAnalysisAgent

        agent = HardwareAnalysisAgent("test", {})

        environment = {'model_size': 2.0}
        agent.observe(environment)

        decision = agent.decide()
        self.assertIsNotNone(decision.method)


if __name__ == "__main__":
    run_tests()
