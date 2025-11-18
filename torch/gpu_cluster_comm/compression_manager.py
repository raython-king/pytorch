"""
Communication Compression Manager
通讯压缩管理器

This module provides compression strategies for reducing communication volume.

本模块提供压缩策略以减少通讯量。
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch

from .types import CompressionStrategy, CompressedTensor
from .utils import get_tensor_size_bytes

logger = logging.getLogger(__name__)


class CompressionManager:
    """
    Manager for communication compression.
    通讯压缩管理器。

    This class provides various compression strategies to reduce the amount
    of data transmitted during communication.

    Attributes:
        default_strategy: Default compression strategy
        enable_error_feedback: Whether to use error feedback
        error_buffer: Buffer for error feedback
    """

    def __init__(
        self,
        default_strategy: CompressionStrategy = CompressionStrategy.NONE,
        enable_error_feedback: bool = True
    ):
        """
        Initialize compression manager.
        初始化压缩管理器。

        Args:
            default_strategy: Default compression strategy
            enable_error_feedback: Enable error feedback for lossy compression
        """
        self.default_strategy = default_strategy
        self.enable_error_feedback = enable_error_feedback

        # Error feedback buffer (for gradient compression)
        self.error_buffer: Dict[str, torch.Tensor] = {}

        # Statistics
        self._stats = {
            'num_compressions': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
        }

    def select_compression(
        self,
        tensor: torch.Tensor,
        target_ratio: float = 0.5
    ) -> CompressionStrategy:
        """
        Select appropriate compression strategy for a tensor.
        为张量选择合适的压缩策略。

        Args:
            tensor: Input tensor
            target_ratio: Target compression ratio (0 to 1)

        Returns:
            Selected compression strategy
        """
        # Factors to consider:
        # 1. Tensor dtype (already compressed?)
        # 2. Tensor size (worthwhile to compress?)
        # 3. Target compression ratio

        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            # Already half precision
            if target_ratio < 0.5:
                # Need more compression
                return CompressionStrategy.INT8
            else:
                return CompressionStrategy.NONE

        elif tensor.dtype == torch.float32:
            # Full precision, can compress
            if target_ratio >= 0.5:
                # Moderate compression: FP16 or BF16
                return CompressionStrategy.FP16
            elif target_ratio >= 0.25:
                # Higher compression: INT8
                return CompressionStrategy.INT8
            else:
                # Very high compression: sparsification
                return CompressionStrategy.TOP_K

        else:
            # Other dtypes, use default
            return self.default_strategy

    def compress_gradient(
        self,
        grad: torch.Tensor,
        strategy: Optional[CompressionStrategy] = None,
        tensor_name: Optional[str] = None
    ) -> CompressedTensor:
        """
        Compress a gradient tensor.
        压缩梯度张量。

        Args:
            grad: Gradient tensor to compress
            strategy: Compression strategy (uses default if None)
            tensor_name: Name for error feedback tracking

        Returns:
            Compressed tensor
        """
        if strategy is None:
            strategy = self.default_strategy

        original_size = get_tensor_size_bytes(grad)

        # Apply error feedback if enabled
        if self.enable_error_feedback and tensor_name:
            if tensor_name in self.error_buffer:
                grad = grad + self.error_buffer[tensor_name]

        # Compress based on strategy
        if strategy == CompressionStrategy.NONE:
            compressed = self._compress_none(grad)

        elif strategy == CompressionStrategy.FP16:
            compressed = self._compress_fp16(grad)

        elif strategy == CompressionStrategy.BF16:
            compressed = self._compress_bf16(grad)

        elif strategy == CompressionStrategy.INT8:
            compressed = self._compress_int8(grad)

        elif strategy == CompressionStrategy.TOP_K:
            compressed = self._compress_topk(grad, k_ratio=0.01)

        elif strategy == CompressionStrategy.RANDOM_K:
            compressed = self._compress_randomk(grad, k_ratio=0.01)

        else:
            # Fallback to no compression
            compressed = self._compress_none(grad)

        # Update error feedback if enabled
        if self.enable_error_feedback and tensor_name:
            decompressed = self.decompress(compressed)
            error = grad - decompressed
            self.error_buffer[tensor_name] = error

        # Update statistics
        compressed_size = compressed.compressed_size
        self._stats['num_compressions'] += 1
        self._stats['total_original_bytes'] += original_size
        self._stats['total_compressed_bytes'] += compressed_size

        logger.debug(
            f"Compressed gradient with {strategy.value}: "
            f"{original_size} -> {compressed_size} bytes "
            f"(ratio: {compressed.compression_ratio:.2f})"
        )

        return compressed

    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress a compressed tensor.
        解压缩压缩张量。

        Args:
            compressed: Compressed tensor

        Returns:
            Decompressed tensor
        """
        strategy = compressed.strategy

        if strategy == CompressionStrategy.NONE:
            return self._decompress_none(compressed)

        elif strategy == CompressionStrategy.FP16:
            return self._decompress_fp16(compressed)

        elif strategy == CompressionStrategy.BF16:
            return self._decompress_bf16(compressed)

        elif strategy == CompressionStrategy.INT8:
            return self._decompress_int8(compressed)

        elif strategy == CompressionStrategy.TOP_K:
            return self._decompress_topk(compressed)

        elif strategy == CompressionStrategy.RANDOM_K:
            return self._decompress_randomk(compressed)

        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

    def error_feedback(
        self,
        original: torch.Tensor,
        compressed: CompressedTensor,
        tensor_name: str
    ) -> torch.Tensor:
        """
        Compute and store error feedback.
        计算并存储误差反馈。

        Args:
            original: Original tensor
            compressed: Compressed tensor
            tensor_name: Name for tracking

        Returns:
            Error tensor
        """
        decompressed = self.decompress(compressed)
        error = original - decompressed

        # Store for next iteration
        self.error_buffer[tensor_name] = error

        return error

    # ========================================================================
    # Compression Implementations / 压缩实现
    # ========================================================================

    def _compress_none(self, tensor: torch.Tensor) -> CompressedTensor:
        """No compression (identity)"""
        return CompressedTensor(
            data=tensor.clone(),
            strategy=CompressionStrategy.NONE,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            compression_ratio=1.0,
        )

    def _compress_fp16(self, tensor: torch.Tensor) -> CompressedTensor:
        """Compress to FP16"""
        compressed_data = tensor.to(torch.float16)

        ratio = get_tensor_size_bytes(compressed_data) / get_tensor_size_bytes(tensor)

        return CompressedTensor(
            data=compressed_data,
            strategy=CompressionStrategy.FP16,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            compression_ratio=ratio,
        )

    def _compress_bf16(self, tensor: torch.Tensor) -> CompressedTensor:
        """Compress to BF16"""
        compressed_data = tensor.to(torch.bfloat16)

        ratio = get_tensor_size_bytes(compressed_data) / get_tensor_size_bytes(tensor)

        return CompressedTensor(
            data=compressed_data,
            strategy=CompressionStrategy.BF16,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            compression_ratio=ratio,
        )

    def _compress_int8(self, tensor: torch.Tensor) -> CompressedTensor:
        """
        Compress to INT8 using quantization.
        使用量化压缩到INT8。
        """
        # Simple linear quantization
        # Find min and max
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        # Quantize to int8 range [-127, 127]
        if max_val - min_val > 0:
            scale = (max_val - min_val) / 254.0
            zero_point = min_val
        else:
            scale = 1.0
            zero_point = 0.0

        quantized = torch.clamp(
            torch.round((tensor - zero_point) / scale),
            -127, 127
        ).to(torch.int8)

        ratio = get_tensor_size_bytes(quantized) / get_tensor_size_bytes(tensor)

        return CompressedTensor(
            data=quantized,
            strategy=CompressionStrategy.INT8,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            metadata={'scale': scale, 'zero_point': zero_point},
            compression_ratio=ratio,
        )

    def _compress_topk(
        self,
        tensor: torch.Tensor,
        k_ratio: float = 0.01
    ) -> CompressedTensor:
        """
        Compress using top-k sparsification.
        使用top-k稀疏化压缩。

        Args:
            tensor: Input tensor
            k_ratio: Ratio of elements to keep

        Returns:
            Compressed tensor with sparse representation
        """
        # Flatten tensor
        flat = tensor.flatten()
        numel = flat.numel()

        # Compute k (number of elements to keep)
        k = max(1, int(numel * k_ratio))

        # Get top-k by absolute value
        abs_flat = flat.abs()
        topk_vals, topk_indices = torch.topk(abs_flat, k)

        # Create sparse representation
        values = flat[topk_indices]

        # Store as indices and values
        compressed_data = torch.stack([
            topk_indices.float(),
            values
        ])

        ratio = get_tensor_size_bytes(compressed_data) / get_tensor_size_bytes(tensor)

        return CompressedTensor(
            data=compressed_data,
            strategy=CompressionStrategy.TOP_K,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            metadata={'k': k, 'numel': numel},
            compression_ratio=ratio,
        )

    def _compress_randomk(
        self,
        tensor: torch.Tensor,
        k_ratio: float = 0.01
    ) -> CompressedTensor:
        """
        Compress using random-k sparsification.
        使用random-k稀疏化压缩。

        Args:
            tensor: Input tensor
            k_ratio: Ratio of elements to keep

        Returns:
            Compressed tensor
        """
        # Flatten tensor
        flat = tensor.flatten()
        numel = flat.numel()

        # Compute k
        k = max(1, int(numel * k_ratio))

        # Random sampling
        perm = torch.randperm(numel, device=tensor.device)
        indices = perm[:k]

        # Get values
        values = flat[indices]

        # Store as indices and values
        compressed_data = torch.stack([
            indices.float(),
            values
        ])

        ratio = get_tensor_size_bytes(compressed_data) / get_tensor_size_bytes(tensor)

        return CompressedTensor(
            data=compressed_data,
            strategy=CompressionStrategy.RANDOM_K,
            original_shape=tensor.shape,
            original_dtype=tensor.dtype,
            metadata={'k': k, 'numel': numel},
            compression_ratio=ratio,
        )

    # ========================================================================
    # Decompression Implementations / 解压缩实现
    # ========================================================================

    def _decompress_none(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress identity"""
        return compressed.data.clone()

    def _decompress_fp16(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress from FP16"""
        return compressed.data.to(compressed.original_dtype)

    def _decompress_bf16(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress from BF16"""
        return compressed.data.to(compressed.original_dtype)

    def _decompress_int8(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress from INT8"""
        scale = compressed.metadata['scale']
        zero_point = compressed.metadata['zero_point']

        # Dequantize
        dequantized = compressed.data.to(torch.float32) * scale + zero_point

        # Reshape and convert to original dtype
        result = dequantized.reshape(compressed.original_shape)
        return result.to(compressed.original_dtype)

    def _decompress_topk(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress from top-k sparse representation"""
        k = compressed.metadata['k']
        numel = compressed.metadata['numel']

        # Extract indices and values
        indices = compressed.data[0, :k].long()
        values = compressed.data[1, :k]

        # Reconstruct sparse tensor
        result = torch.zeros(numel, dtype=compressed.original_dtype,
                             device=compressed.data.device)
        result[indices] = values

        # Reshape
        return result.reshape(compressed.original_shape)

    def _decompress_randomk(self, compressed: CompressedTensor) -> torch.Tensor:
        """Decompress from random-k sparse representation"""
        return self._decompress_topk(compressed)  # Same as top-k

    # ========================================================================
    # Utility Methods / 工具方法
    # ========================================================================

    def estimate_compression_ratio(
        self,
        tensor: torch.Tensor,
        strategy: CompressionStrategy
    ) -> float:
        """
        Estimate compression ratio without actually compressing.
        在不实际压缩的情况下估计压缩比。

        Args:
            tensor: Input tensor
            strategy: Compression strategy

        Returns:
            Estimated compression ratio
        """
        if strategy == CompressionStrategy.NONE:
            return 1.0

        elif strategy == CompressionStrategy.FP16:
            if tensor.dtype == torch.float32:
                return 0.5
            else:
                return 1.0

        elif strategy == CompressionStrategy.BF16:
            if tensor.dtype == torch.float32:
                return 0.5
            else:
                return 1.0

        elif strategy == CompressionStrategy.INT8:
            if tensor.dtype == torch.float32:
                return 0.25
            elif tensor.dtype == torch.float16:
                return 0.5
            else:
                return 1.0

        elif strategy in [CompressionStrategy.TOP_K, CompressionStrategy.RANDOM_K]:
            # Sparse: depends on k_ratio (assume 1%)
            k_ratio = 0.01
            # Need to store indices (int32) + values
            # Approximate: k * (4 + element_size) / total_size
            element_size = tensor.element_size()
            total_size = tensor.numel() * element_size
            sparse_size = tensor.numel() * k_ratio * (4 + element_size)
            return sparse_size / total_size

        else:
            return 1.0

    def compute_compression_benefit(
        self,
        tensor_size: int,
        bandwidth_gbps: float,
        strategy: CompressionStrategy
    ) -> Dict[str, float]:
        """
        Compute the benefit of compression.
        计算压缩的收益。

        Args:
            tensor_size: Original tensor size in bytes
            bandwidth_gbps: Network bandwidth in GB/s
            strategy: Compression strategy

        Returns:
            Dictionary with benefit metrics
        """
        # Estimate compression ratio
        dummy_tensor = torch.zeros(tensor_size // 4, dtype=torch.float32)
        ratio = self.estimate_compression_ratio(dummy_tensor, strategy)

        compressed_size = int(tensor_size * ratio)

        # Communication time without compression
        time_no_compress = (tensor_size / (bandwidth_gbps * 1e9)) * 1e6  # microseconds

        # Communication time with compression
        time_with_compress = (compressed_size / (bandwidth_gbps * 1e9)) * 1e6

        # Compression/decompression overhead (estimated)
        # Assume 10 us per MB
        compress_overhead = (tensor_size / (1024 * 1024)) * 10.0
        decompress_overhead = (compressed_size / (1024 * 1024)) * 10.0

        total_time_compress = (
            time_with_compress + compress_overhead + decompress_overhead
        )

        # Speedup
        if total_time_compress > 0:
            speedup = time_no_compress / total_time_compress
        else:
            speedup = 1.0

        return {
            'compression_ratio': ratio,
            'bytes_saved': tensor_size - compressed_size,
            'time_no_compress': time_no_compress,
            'time_with_compress': total_time_compress,
            'speedup': speedup,
            'is_beneficial': speedup > 1.0,
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get compression statistics.
        获取压缩统计信息。

        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()

        # Compute average compression ratio
        if stats['total_original_bytes'] > 0:
            stats['avg_compression_ratio'] = (
                stats['total_compressed_bytes'] / stats['total_original_bytes']
            )
        else:
            stats['avg_compression_ratio'] = 1.0

        return stats

    def reset_statistics(self) -> None:
        """Reset statistics"""
        self._stats = {
            'num_compressions': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
        }

    def clear_error_buffer(self) -> None:
        """Clear error feedback buffer"""
        self.error_buffer.clear()
