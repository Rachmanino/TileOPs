# Ascend 950 GEMM candidates

This contribution prepares 10 recovered TileLang BF16/NT GEMM candidates for
`GemmFwdOp`: `D = A @ B.T`, contiguous BF16 inputs and output, FP32 accumulation.
The intended dispatch covers 10 of the current 20 public manifest cases. The
historical archive contains 11 candidates, but W7 is deferred: its original source
is 905,929 bytes (above the 500 KiB source-file check) and its measurement has
material drift. W7 is not in the public dispatch, and no lint exemption is added.
The other nine public cases have no candidate in this historical set. Refs [#8](https://github.com/yyttt6/TileOPs/issues/8);
this partial contribution does not close that issue.

## Evidence status

The table records historical measurements of the original kernels on a **950DT
with 36 Cube cores**, using a nominal dense BF16 peak of **486 TFLOP/s**. It does
not use measurements from the 28-Cube 950PR. Geometry and some buffer depths were
manually aligned with DeepGEMM, then schedule variables were solved. These are
useful measured candidates, not evidence of successful unattended autotuning.

The recovered source bytes match both the historical input manifest and runtime
binding hashes. Schedule hashes also match those bindings. Historical correctness
checks used seeds 0, 1 and 7 and were bitwise equal to DeepGEMM. Archived checks
before, during and after every run reported no foreign process; final memory usage
returned below the historical idle ceiling. These checks have been replayed from
receipts, not rerun on hardware for this contribution.

**No new runtime or public performance validation has been completed.** Historical
runs used 10 warmups and 30 timed invocations per arm in a round-robin profiler
capture, with a 2×L2 zeroing pass before each timed arm. They report medians of the
historical device-kernel duration metric. These are not interchangeable with the public performance protocol. The
[current roadmap](https://github.com/yyttt6/TileOPs/issues/1) specifies the
platform's default configuration and says the updated performance entry point and
baseline package are still awaiting publication. The older locked-clock protocol
in the development guide is not the next acceptance target. Performance remains
pending until the maintained public protocol and matching baselines are available.
The old environment's version label alone does not establish that it matches the
current official TileLang release. Modified ports must be compiled and checked
again; their performance cannot inherit the original source's receipt.

## Historical 950DT measurements

Within each paired column, values are **candidate / DeepGEMM**. Tile dimensions
are M×N×K. Buffer depths list `L1 A/B, L0 A/B` separately for each main or tail
tile class. Peak utilization uses the same 486 TFLOP/s denominator for both arms.
Throughput ratio is DeepGEMM median latency divided by candidate median latency.
Drift is the second-half median divided by the first-half median, minus one.

| ID  | M×N×K           | Outer tile  | Buffer depths by tile class          | Median μs             | TFLOP/s         | Peak utilization | Throughput ratio | Drift             |
| --- | --------------- | ----------- | ------------------------------------ | --------------------- | --------------- | ---------------- | ---------------- | ----------------- |
| W1  | 128×7168×2048   | 128×208×128 | 128×208: 1/2, 2/2; 128×96: 2/1, 2/2  | 16.2340 / 16.6085     | 231.50 / 226.28 | 47.63% / 46.56%  | 102.31%          | -0.02% / +0.21%   |
| W2  | 128×2112×7168   | 128×64×256  | 128×64: 2/1, 2/2                     | 28.8770 / 24.9165     | 134.21 / 155.54 | 27.61% / 32.00%  | 86.28%           | -0.73% / -0.60%   |
| W3  | 4096×4096×7168  | 224×256×128 | 224×256: 4/4, 2/3; 64×256: 4/4, 2/3  | 545.3095 / 525.3320   | 441.07 / 457.84 | 90.75% / 94.21%  | 96.34%           | +0.12% / +0.04%   |
| W5  | 4096×7168×2048  | 256×160×128 | 256×160: 4/4, 2/3; 256×128: 4/4, 2/4 | 312.2555 / 274.9595   | 385.13 / 437.37 | 79.24% / 89.99%  | 88.06%           | -0.16% / -0.14%   |
| W6  | 4096×2112×7168  | 256×240×128 | 256×240: 4/4, 3/2; 256×192: 4/4, 3/2 | 278.1135 / 267.0265   | 445.92 / 464.44 | 91.75% / 95.56%  | 96.01%           | -0.03% / +0.02%   |
| W7  | 4096×7168×16384 | 256×160×128 | 256×160: 4/4, 2/3; 256×128: 4/4, 2/4 | 2203.6515 / 2197.3955 | 436.58 / 437.82 | 89.83% / 90.09%  | 99.72%           | +24.21% / +23.68% |
| W8  | 16×4096×7168    | 16×128×256  | 16×128: 1/2, 1/2                     | 24.6645 / 22.1395     | 38.09 / 42.44   | 7.84% / 8.73%    | 89.76%           | -1.76% / +0.34%   |
| W9  | 32×4096×7168    | 32×128×256  | 32×128: 1/2, 2/2                     | 26.7950 / 24.5230     | 70.13 / 76.62   | 14.43% / 15.77%  | 91.52%           | -0.57% / -0.33%   |
| W10 | 64×7168×2048    | 64×208×128  | 64×208: 1/2, 2/2; 64×96: 2/2, 2/2    | 13.9270 / 14.4720     | 134.92 / 129.84 | 27.76% / 26.72%  | 103.91%          | +0.16% / -0.74%   |
| W11 | 96×2112×7168    | 96×64×256   | 96×64: 2/1, 2/2                      | 28.5845 / 21.3650     | 101.69 / 136.05 | 20.92% / 27.99%  | 74.74%           | +0.86% / -0.29%   |
| W14 | 4096×24576×1536 | 256×256×128 | 256×256: 4/4, 2/3                    | 736.4225 / 645.6105   | 419.92 / 478.98 | 86.40% / 98.56%  | 87.67%           | -0.28% / +0.00%   |

**W7 has material drift:** candidate latency increased 24.21% between halves,
DeepGEMM increased 23.68%, and the original TileLang example increased 28.08%.
Its 99.72% aggregate throughput ratio is a historical diagnostic and requires a
stable rerun before a performance claim. W7 is retained here as historical data
only and is not imported. Six aggregate ratios exceed 90%; excluding deferred W7,
five imported candidates exceed 90% without that material drift. These numbers are neither a current acceptance result nor a
performance guarantee for another device, compiler or wrapper.

## Case mapping and original source identity

Full SHA256 values identify the recovered original source bytes. Formatting,
factory renaming or API adaptations change those hashes. The sanitized
[provenance data](gemm-provenance.json) retains the numeric distributions, MAD
tiles, per-class depths, manifest revision and source hashes without private
runtime paths or raw receipts.

| Historical ID | Public case ID                  | Import status             | Original source SHA256                                             |
| ------------- | ------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| W1            | `ds-v3-decode-down-bfloat16`    | Prepared for import       | `6b620386b457b3c682c264bd3e7d9d8d9057ad0b38533b8f7fd8780f84b25907` |
| W2            | `ds-v3-decode-qkv-a-bfloat16`   | Prepared for import       | `b1612bb04be34195b45c6e55e9fe6379899957203cfb62e38d95783cf3fba328` |
| W3            | `ds-v3-prefill-mlp-up-bfloat16` | Prepared for import       | `51f2c7f967eaf9d32ea2be6448b78fed76b37640158cb78946b7e2b833b326f3` |
| W5            | `ds-v3-prefill-down-bfloat16`   | Prepared for import       | `0e3b88e62bb399887b21ab60574ab6dcd03d94c97e516b61f6a14af2b30792db` |
| W6            | `ds-v3-prefill-qkv-a-bfloat16`  | Prepared for import       | `69bcd9514e8a36cb5a557ff4c4c8a7f10866071668f2b5fb2e853d0dde299b01` |
| W7            | `ds-v3-prefill-o-proj-bfloat16` | Deferred; historical only | `ff0cb96acd59fba3835b4440431bda588fc9878ba4e81c2506aad9960f37b64a` |
| W8            | `ds-v3-m16-mlp-up-bfloat16`     | Prepared for import       | `80f8005c5bff1507bb633973dceb748572dd6b4b7945393530744437d8edce08` |
| W9            | `ds-v3-m32-mlp-up-bfloat16`     | Prepared for import       | `4ef27e4ee5ae1d0954343f3f0e39df5a47e6e31de44a47dc34ba00fc8c0e11f0` |
| W10           | `ds-v3-m64-down-bfloat16`       | Prepared for import       | `844e450f526e9b20736db8d9d3585cba2b3d115f1a468b41fd3fc5724982cc5c` |
| W11           | `ds-v3-m96-qkv-a-bfloat16`      | Prepared for import       | `75a9899ccbe278272f0c55834f048d208ec2bf9fe2a122437bafbbae539f7097` |
| W14           | `ds-v3-prefill-q-b-bfloat16`    | Prepared for import       | `642649ac67e639dffa0fff91ac71fea00a693893386e899bad6c005abbc394da` |

Ten public cases remain outside the prepared dispatch: deferred
`ds-v3-prefill-o-proj-bfloat16` (W7), plus the following nine cases without a
candidate in this historical set:

- `square-1k-float16`
- `square-1k-bfloat16`
- `ds-v3-prefill-mlp-up-float16`
- `ds-v3-m1-qkv-a-bfloat16`
- `ds-v3-m2-qkv-a-bfloat16`
- `qwen3-8b-gate-up-decode-bfloat16`
- `qwen3-8b-gate-up-b32-bfloat16`
- `qwen3-8b-down-prefill-bfloat16`
- `vit-l16-in21k-head-bfloat16`

## Revalidate the imported subset

In the official environment, the current shared launcher can check correctness
without clock locking. Use a fresh output directory and the site's shared guard:

```bash
python scripts/benchmark_ascend.py \
  --owner developer/gemm \
  --run-dir ./results/gemm-verify-001 \
  --op GemmFwdOp \
  --case-regex '^(ds-v3-decode-down-bfloat16|ds-v3-decode-qkv-a-bfloat16|ds-v3-prefill-mlp-up-bfloat16|ds-v3-prefill-down-bfloat16|ds-v3-prefill-qkv-a-bfloat16|ds-v3-m16-mlp-up-bfloat16|ds-v3-m32-mlp-up-bfloat16|ds-v3-m64-down-bfloat16|ds-v3-m96-qkv-a-bfloat16|ds-v3-prefill-q-b-bfloat16)$' \
  --verify-only
```

This explicit filter selects the ten prepared cases; it is a subset check, not
completion of the 20-case task. Require all ten cases and the final guard receipt
to pass. This command has not been executed for the port and produces no performance
acceptance. Full coverage still requires W7 and the nine cases listed above.

Remaining work for #8 includes these cases and fresh correctness/benchmark runs
through the shared launcher. Historical correctness uses a different harness;
it cannot replace the public workload reference and numerical policy.
