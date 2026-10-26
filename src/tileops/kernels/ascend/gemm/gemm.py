"""Shape-specialized BF16 NT GEMM schedules for Ascend 950."""

import functools
import importlib
from typing import Optional

import torch

import tileops.backend
from tileops.kernels.ascend.gemm.persistent import KERNEL_MODULES
from tileops.kernels.gemm import GemmCall, GemmFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


@functools.lru_cache(maxsize=32)
def _make_gemm_kernel(m: int, n: int, k: int, device_index: int):
    name = torch.npu.get_device_name(device_index)
    if "Ascend950" not in name:
        raise RuntimeError(f"These GEMM schedules require Ascend 950, got {name}")
    module = importlib.import_module(
        f"tileops.kernels.ascend.gemm.persistent.{KERNEL_MODULES[m, n, k]}"
    )
    return module.compile_kernel()


class AscendGemmKernel(Kernel, GemmFwdInterface):
    """Persistent BF16 ``a @ b.T`` with explicit transfer and Cube events."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GemmCall) -> Optional[str]:
        if call.dtype != torch.bfloat16:
            return "requires bfloat16 operands"
        if call.trans_a or not call.trans_b:
            return "requires NT layout (trans_a=False, trans_b=True)"
        if (call.m, call.n, call.k) not in KERNEL_MODULES:
            return f"no specialized schedule for M={call.m}, N={call.n}, K={call.k}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: GemmCall):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        self._compiled_fn = _make_gemm_kernel(call.m, call.n, call.k, device_index)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._compiled_fn(a, b)


tileops.backend.register_kernel_type("GemmFwdOp", "ascend", AscendGemmKernel)
