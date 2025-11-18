"""
Backward Compatibility Module for GPU Cluster Communication Optimization

This module ensures backward compatibility with different PyTorch versions
and provides fallback mechanisms for unsupported configurations.

Key features:
- Version compatibility checking
- Graceful fallback to native implementations
- Behavior validation
- Feature detection
"""

import logging
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from packaging import version

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CompatibilityError(Exception):
    """Raised when compatibility check fails."""
    pass


class CompatibilityWarning(UserWarning):
    """Warning for compatibility issues that don't prevent operation."""
    pass


class VersionInfo:
    """PyTorch version information."""

    def __init__(self):
        self.pytorch_version = torch.__version__
        self.cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        self.cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        self.nccl_version = torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else None
        self.python_version = sys.version_info

    def __repr__(self):
        return (
            f"VersionInfo(\n"
            f"  PyTorch: {self.pytorch_version}\n"
            f"  CUDA: {self.cuda_version}\n"
            f"  cuDNN: {self.cudnn_version}\n"
            f"  NCCL: {self.nccl_version}\n"
            f"  Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}\n"
            f")"
        )


class BackwardCompatibility:
    """Ensure backward compatibility with different PyTorch versions."""

    # Minimum supported versions
    MIN_PYTORCH_VERSION = "1.10.0"
    MIN_CUDA_VERSION = "10.2"
    MIN_PYTHON_VERSION = (3, 7)

    # Recommended versions for best performance
    RECOMMENDED_PYTORCH_VERSION = "2.0.0"
    RECOMMENDED_CUDA_VERSION = "11.8"

    def __init__(self):
        self.version_info = VersionInfo()
        self._compatibility_cache = {}
        self._feature_support = {}
        self._check_basic_compatibility()

    def _check_basic_compatibility(self):
        """Check basic compatibility requirements."""
        # Check Python version
        if sys.version_info < self.MIN_PYTHON_VERSION:
            raise CompatibilityError(
                f"Python {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}+ required, "
                f"got {sys.version_info.major}.{sys.version_info.minor}"
            )

        # Check PyTorch version
        pytorch_version = version.parse(self.version_info.pytorch_version.split('+')[0])
        min_pytorch = version.parse(self.MIN_PYTORCH_VERSION)

        if pytorch_version < min_pytorch:
            raise CompatibilityError(
                f"PyTorch {self.MIN_PYTORCH_VERSION}+ required, "
                f"got {self.version_info.pytorch_version}"
            )

        # Warn if not using recommended version
        recommended = version.parse(self.RECOMMENDED_PYTORCH_VERSION)
        if pytorch_version < recommended:
            warnings.warn(
                f"PyTorch {self.RECOMMENDED_PYTORCH_VERSION}+ recommended for best performance, "
                f"got {self.version_info.pytorch_version}",
                CompatibilityWarning
            )

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise CompatibilityError("CUDA not available - GPU cluster optimization requires CUDA")

        # Check distributed support
        if not dist.is_available():
            raise CompatibilityError("torch.distributed not available")

        logger.info(f"Compatibility check passed: {self.version_info}")

    def check_pytorch_version(self, min_version: str) -> bool:
        """Check if PyTorch version meets minimum requirement.

        Args:
            min_version: Minimum version string (e.g., "1.10.0")

        Returns:
            True if version is sufficient
        """
        current = version.parse(self.version_info.pytorch_version.split('+')[0])
        required = version.parse(min_version)
        return current >= required

    def check_feature_support(self, feature: str) -> bool:
        """Check if a specific feature is supported.

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        if feature in self._feature_support:
            return self._feature_support[feature]

        # Check various features
        supported = False

        if feature == "nccl":
            supported = hasattr(torch.cuda, 'nccl')

        elif feature == "ddp_comm_hook":
            supported = self.check_pytorch_version("1.8.0")

        elif feature == "fsdp":
            supported = self.check_pytorch_version("1.11.0")

        elif feature == "cuda_graphs":
            supported = self.check_pytorch_version("1.10.0") and \
                       self.version_info.cuda_version is not None

        elif feature == "nvlink":
            try:
                # Check if we can query NVLink status
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    # Simple heuristic: try to check device properties
                    supported = True
                else:
                    supported = False
            except:
                supported = False

        elif feature == "async_allreduce":
            supported = self.check_pytorch_version("1.6.0")

        elif feature == "process_group":
            supported = hasattr(dist, 'new_group')

        else:
            logger.warning(f"Unknown feature: {feature}")
            supported = False

        self._feature_support[feature] = supported
        return supported

    def get_supported_features(self) -> List[str]:
        """Get list of all supported features."""
        features = [
            "nccl", "ddp_comm_hook", "fsdp", "cuda_graphs",
            "nvlink", "async_allreduce", "process_group"
        ]
        return [f for f in features if self.check_feature_support(f)]

    def fallback_to_native(self, operation: str, error: Exception,
                          fallback_func: Callable, *args, **kwargs) -> Any:
        """Fallback to native implementation on error.

        Args:
            operation: Name of the operation
            error: Exception that triggered fallback
            fallback_func: Native function to fall back to
            *args, **kwargs: Arguments to pass to fallback function

        Returns:
            Result of fallback function
        """
        logger.warning(
            f"Falling back to native implementation for {operation} due to error: {error}"
        )

        try:
            result = fallback_func(*args, **kwargs)
            logger.info(f"Fallback successful for {operation}")
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {operation}: {fallback_error}")
            raise

    def validate_behavior(self, operation: str,
                         native_func: Callable,
                         optimized_func: Callable,
                         test_inputs: List[Tuple],
                         tolerance: float = 1e-5) -> bool:
        """Validate that optimized version produces same results as native.

        Args:
            operation: Name of operation being tested
            native_func: Native implementation
            optimized_func: Optimized implementation
            test_inputs: List of input tuples to test
            tolerance: Numerical tolerance for comparison

        Returns:
            True if all tests pass
        """
        logger.info(f"Validating behavior for {operation} with {len(test_inputs)} test cases")

        all_passed = True
        for i, inputs in enumerate(test_inputs):
            try:
                # Run native version
                if torch.is_tensor(inputs[0]):
                    native_input = inputs[0].clone()
                    optimized_input = inputs[0].clone()
                else:
                    native_input = inputs[0]
                    optimized_input = inputs[0]

                native_result = native_func(native_input, *inputs[1:])
                optimized_result = optimized_func(optimized_input, *inputs[1:])

                # Compare results
                if torch.is_tensor(native_result):
                    if not torch.allclose(native_result, optimized_result,
                                         rtol=tolerance, atol=tolerance):
                        logger.error(
                            f"Test case {i} failed: results differ\n"
                            f"Max diff: {torch.max(torch.abs(native_result - optimized_result))}"
                        )
                        all_passed = False
                else:
                    if native_result != optimized_result:
                        logger.error(f"Test case {i} failed: {native_result} != {optimized_result}")
                        all_passed = False

            except Exception as e:
                logger.error(f"Test case {i} raised exception: {e}")
                all_passed = False

        if all_passed:
            logger.info(f"All validation tests passed for {operation}")
        else:
            logger.error(f"Some validation tests failed for {operation}")

        return all_passed

    def get_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report.

        Returns:
            Dictionary containing compatibility information
        """
        report = {
            'version_info': {
                'pytorch': self.version_info.pytorch_version,
                'cuda': self.version_info.cuda_version,
                'cudnn': self.version_info.cudnn_version,
                'nccl': self.version_info.nccl_version,
                'python': f"{self.version_info.python_version.major}."
                         f"{self.version_info.python_version.minor}."
                         f"{self.version_info.python_version.micro}",
            },
            'compatibility': {
                'meets_minimum': self.check_pytorch_version(self.MIN_PYTORCH_VERSION),
                'recommended': self.check_pytorch_version(self.RECOMMENDED_PYTORCH_VERSION),
            },
            'features': {
                feature: self.check_feature_support(feature)
                for feature in [
                    'nccl', 'ddp_comm_hook', 'fsdp', 'cuda_graphs',
                    'nvlink', 'async_allreduce', 'process_group'
                ]
            },
            'hardware': {
                'cuda_available': torch.cuda.is_available(),
                'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'distributed_available': dist.is_available(),
            },
        }

        return report

    def print_compatibility_report(self):
        """Print human-readable compatibility report."""
        report = self.get_compatibility_report()

        print("\n" + "=" * 70)
        print("GPU Cluster Communication Compatibility Report")
        print("=" * 70)

        print("\nVersion Information:")
        for key, value in report['version_info'].items():
            print(f"  {key.upper()}: {value}")

        print("\nCompatibility Status:")
        for key, value in report['compatibility'].items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")

        print("\nFeature Support:")
        for feature, supported in report['features'].items():
            status = "✓" if supported else "✗"
            print(f"  {status} {feature}: {supported}")

        print("\nHardware:")
        for key, value in report['hardware'].items():
            print(f"  {key}: {value}")

        print("=" * 70 + "\n")


class SafetyWrapper:
    """Wrapper that provides safe execution with automatic fallback."""

    def __init__(self, optimized_func: Callable, native_func: Callable,
                 operation_name: str):
        self.optimized_func = optimized_func
        self.native_func = native_func
        self.operation_name = operation_name
        self.failure_count = 0
        self.max_failures = 3
        self.permanently_disabled = False

    def __call__(self, *args, **kwargs):
        """Execute with automatic fallback on failure."""
        if self.permanently_disabled:
            return self.native_func(*args, **kwargs)

        try:
            return self.optimized_func(*args, **kwargs)
        except Exception as e:
            self.failure_count += 1
            logger.warning(
                f"Optimized {self.operation_name} failed ({self.failure_count}/{self.max_failures}): {e}"
            )

            if self.failure_count >= self.max_failures:
                self.permanently_disabled = True
                logger.error(
                    f"Permanently disabling optimized {self.operation_name} "
                    f"after {self.failure_count} failures"
                )

            # Fallback to native
            return self.native_func(*args, **kwargs)

    def reset(self):
        """Reset failure counter."""
        self.failure_count = 0
        self.permanently_disabled = False


class FeatureGate:
    """Gate features based on compatibility."""

    def __init__(self, compatibility: BackwardCompatibility):
        self.compatibility = compatibility
        self._gates = {}

    def require_feature(self, feature: str, error_msg: Optional[str] = None):
        """Require a feature to be supported, raise error if not.

        Args:
            feature: Feature name
            error_msg: Optional custom error message
        """
        if not self.compatibility.check_feature_support(feature):
            msg = error_msg or f"Required feature '{feature}' not supported"
            raise CompatibilityError(msg)

    def gate(self, feature: str, func: Callable, fallback: Optional[Callable] = None):
        """Gate a function based on feature support.

        Args:
            feature: Feature to check
            func: Function to call if feature is supported
            fallback: Function to call if feature is not supported

        Returns:
            Gated function
        """
        def gated_func(*args, **kwargs):
            if self.compatibility.check_feature_support(feature):
                return func(*args, **kwargs)
            elif fallback is not None:
                logger.warning(f"Feature '{feature}' not supported, using fallback")
                return fallback(*args, **kwargs)
            else:
                raise CompatibilityError(f"Feature '{feature}' not supported and no fallback provided")

        return gated_func


# Global compatibility checker
_global_compatibility = None


def get_compatibility() -> BackwardCompatibility:
    """Get global compatibility checker instance."""
    global _global_compatibility
    if _global_compatibility is None:
        _global_compatibility = BackwardCompatibility()
    return _global_compatibility


def check_compatibility() -> bool:
    """Check if system is compatible.

    Returns:
        True if compatible
    """
    try:
        compat = get_compatibility()
        return compat.check_pytorch_version(BackwardCompatibility.MIN_PYTORCH_VERSION)
    except CompatibilityError:
        return False


def print_compatibility_report():
    """Print compatibility report."""
    compat = get_compatibility()
    compat.print_compatibility_report()
