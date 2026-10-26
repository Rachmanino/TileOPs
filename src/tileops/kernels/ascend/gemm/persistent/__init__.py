"""Frozen Ascend 950 persistent matrix multiplication specializations."""

# The dispatch wrapper uses these identities without importing TileLang kernels.
KERNEL_MODULES: dict[tuple[int, int, int], str] = {
    (128, 7168, 2048): "bf16_nt_128_7168_2048_dt",
    (128, 2112, 7168): "bf16_nt_128_2112_7168_dt",
    (4096, 4096, 7168): "bf16_nt_4096_4096_7168_dt",
    (4096, 7168, 2048): "bf16_nt_4096_7168_2048_dt",
    (4096, 2112, 7168): "bf16_nt_4096_2112_7168_dt",
    (16, 4096, 7168): "bf16_nt_16_4096_7168_dt",
    (32, 4096, 7168): "bf16_nt_32_4096_7168_dt",
    (64, 7168, 2048): "bf16_nt_64_7168_2048_dt",
    (96, 2112, 7168): "bf16_nt_96_2112_7168_dt",
    (4096, 24576, 1536): "bf16_nt_4096_24576_1536_dt",
}

__all__ = ["KERNEL_MODULES"]
