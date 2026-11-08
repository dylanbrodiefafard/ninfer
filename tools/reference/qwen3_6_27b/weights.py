"""Vision tensor materialization over the typed artifact binding."""

from __future__ import annotations

import warnings

import torch

from tools.artifact import decode_direct, dequantize_row_split

from .bindings import CONTIGUOUS, PhysicalBlock, VisionArtifactBinding


class WeightStore:
    """Materialize Vision blocks without persistent-name lookup in the hot path.

    Direct (contiguous) controls are decoded on first use and retained until
    :meth:`close`; they are small and every encoder layer reads them. Quantized
    matrices are streamed from the artifact mapping and dequantized on every
    request, because the Vision schedule consumes each matrix once per encode.
    """

    def __init__(
        self,
        binding: VisionArtifactBinding,
        device: torch.device | str,
        *,
        compile_codec: bool,
    ):
        self.binding = binding
        self.device = torch.device(device)
        self.compile_codec = compile_codec
        self._decoded: dict[int, torch.Tensor] = {}

    @staticmethod
    def _readonly_tensor(payload: memoryview) -> torch.Tensor:
        # PyTorch warns because an mmap opened read-only cannot be mutated.  The
        # codec only reads this tensor, so a zero-copy host view is intentional.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            return torch.frombuffer(payload, dtype=torch.uint8)

    def _source(self, block: PhysicalBlock) -> torch.Tensor:
        host = self._readonly_tensor(self.binding.payload(block))
        return host if self.device.type == "cpu" else host.to(self.device)

    def tensor(self, block: PhysicalBlock) -> torch.Tensor:
        if block.layout != CONTIGUOUS:
            return dequantize_row_split(
                self._source(block),
                block.format,
                block.shape,
                device=self.device,
                compiled=self.compile_codec,
            )
        cached = self._decoded.get(block.tensor_id)
        if cached is not None:
            return cached
        decoded = decode_direct(self._source(block), block.format, block.shape, device=self.device)
        # A CPU result may alias the artifact mapping through torch.frombuffer.
        # The retained control owns independent storage so it outlives the mapping.
        if self.device.type == "cpu":
            decoded = decoded.clone()
        self._decoded[block.tensor_id] = decoded
        return decoded

    def close(self) -> None:
        self._decoded.clear()


__all__ = ["WeightStore"]
