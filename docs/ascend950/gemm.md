# Ascend 950 BF16 GEMM

Ten BF16/NT specializations serve `GemmFwdOp(trans_b=True)` with FP32 accumulation.
Unsupported shapes, dtypes and layouts are refused. This covers 10 of the 20 cases in
[issue #8](https://github.com/yyttt6/TileOPs/issues/8); the issue remains open.

## Implementation and validation

Repeated pipeline regions use ordinary `for ... in range(...)`. Prologue, drain,
event IDs and buffer slots retain the original ordering. Ten kernel files total
**10,845 lines, down from 25,520**. The short-K 4096×24576×1536 kernel has no complete
repeated ring cycle and remains explicit.

Expanding the loops reproduces each original module AST, after normalizing affine
tensor addresses. The proof includes flags, clear-accumulation, allocations, launch,
ABI and compiler options. [Source hashes](gemm-provenance.json) identify both versions.
**Compilation, correctness and performance on the official device environment remain pending.**

## Historical performance

Original kernels on **950DT, 36 Cube, nominal BF16 peak 486 TFLOP/s**. Geometry and
some depths were manually aligned before solving. Timings are medians of 30 interleaved
device-kernel samples after 10 warmups, with L2 clearing. The historical TileLang build
was labelled 0.1.15; equivalence to the official release has not been established.
These are historical results, not measurements of the compact ports.

| M×N×K           | Original μs | DeepGEMM μs | Original TFLOP/s | Peak utilization | Original/DG throughput |
| --------------- | ----------: | ----------: | ---------------: | ---------------: | ---------------------: |
| 128×7168×2048   |      16.234 |      16.608 |           231.50 |           47.63% |                102.31% |
| 128×2112×7168   |      28.877 |      24.916 |           134.21 |           27.61% |                 86.28% |
| 4096×4096×7168  |     545.309 |     525.332 |           441.07 |           90.75% |                 96.34% |
| 4096×7168×2048  |     312.255 |     274.959 |           385.13 |           79.24% |                 88.06% |
| 4096×2112×7168  |     278.114 |     267.026 |           445.92 |           91.75% |                 96.01% |
| 16×4096×7168    |      24.665 |      22.139 |            38.09 |            7.84% |                 89.76% |
| 32×4096×7168    |      26.795 |      24.523 |            70.13 |           14.43% |                 91.52% |
| 64×7168×2048    |      13.927 |      14.472 |           134.92 |           27.76% |                103.91% |
| 96×2112×7168    |      28.584 |      21.365 |           101.69 |           20.92% |                 74.74% |
| 4096×24576×1536 |     736.423 |     645.611 |           419.92 |           86.40% |                 87.67% |

Five original kernels exceeded 90% of DeepGEMM throughput. The separate historical
4096×7168×16384 candidate is not imported: its run drifted about 24%.

## Reproduce correctness

Use the official environment, an idle device and a fresh result directory. The existing
launcher acquires the shared guard; this checks only the ten supported cases.

```bash
python scripts/benchmark_ascend.py \
  --owner developer/gemm --run-dir ./results/gemm-verify-001 \
  --op GemmFwdOp --timeout 600 \
  --case-regex '^(ds-v3-decode-down-bfloat16|ds-v3-decode-qkv-a-bfloat16|ds-v3-prefill-mlp-up-bfloat16|ds-v3-prefill-down-bfloat16|ds-v3-prefill-qkv-a-bfloat16|ds-v3-m16-mlp-up-bfloat16|ds-v3-m32-mlp-up-bfloat16|ds-v3-m64-down-bfloat16|ds-v3-m96-qkv-a-bfloat16|ds-v3-prefill-q-b-bfloat16)$' \
  --verify-only
```

Formal performance revalidation awaits the updated default-platform entry point and
baseline described in [issue #1](https://github.com/yyttt6/TileOPs/issues/1).
