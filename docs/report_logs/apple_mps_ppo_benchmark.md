# Apple MPS PPO Device Support and Benchmark

Implementation date: **2026-08-09 (Asia/Karachi)**

Benchmark status: **COMPLETED — CPU and MPS both completed successfully; the
measured result supports retaining CPU as the production default.**

## Objective and scope

This focused enhancement adds explicit `cpu`, `mps`, and `auto` runtime-device
support to the existing single-symbol Stable-Baselines3 PPO trainer and defines
an apples-to-apples CPU-versus-Apple-MPS benchmark. It does not change the
versioned PPO research hyperparameters, environment trading semantics,
candidate-selection rules, or RL data contract.

The benchmark is an infrastructure measurement, not a model-quality or
profitability experiment. It trains one in-memory model per device from the
same canonical TRAIN partition. It does not evaluate VALIDATION or TEST, save a
production model, append to the model registry, or promote an artifact.

## Jupyter, PyTorch, and MPS

Jupyter does not have a notebook-wide switch that transparently moves arbitrary
Python work to the GPU. A notebook kernel is a normal Python process. NumPy,
pandas, Gymnasium environment logic, and ordinary Python code continue to run
on the CPU. PyTorch tensors and modules run on an accelerator only when the
application explicitly places them on an accelerator device.

For this project, the trainer resolves the requested device and passes the
concrete result directly to Stable-Baselines3 PPO. Stable-Baselines3 then moves
the PPO policy and its PyTorch tensors to that device. The Gymnasium environment
and market-data frame remain CPU-side, as expected.

PyTorch distinguishes two MPS capability checks:

- `torch.backends.mps.is_built()` reports whether the installed PyTorch binary
  was compiled with MPS support.
- `torch.backends.mps.is_available()` reports whether MPS can actually be used
  by the current process on the current machine.

Being built is not sufficient by itself. An explicit MPS request is accepted
only when MPS is available at runtime.

## Device resolution contract

The production PPO default remains conservatively set to **`cpu`** until the
controlled benchmark demonstrates that MPS is materially faster and stable.

| Requested device | Resolution behavior |
|---|---|
| `cpu` | Always resolves to `cpu`. |
| `mps` | Resolves to `mps` only when PyTorch reports MPS available; otherwise fails clearly without falling back. |
| `auto` | Resolves to `mps` when available and otherwise resolves explicitly to `cpu`. The result records both the request and the resolution. |

Stable-Baselines3 2.9.0's own `device="auto"` behavior does **not** select Apple
MPS: it attempts CUDA and otherwise uses CPU. The project therefore resolves
`auto` centrally before PPO construction and passes either `cpu` or `mps`
explicitly. It does not rely on Stable-Baselines3 AUTO behavior.

After constructing PPO, the trainer verifies both the device reported by the
SB3 model and the device of the policy's PyTorch parameters. Equivalent MPS
spellings such as `mps` and `mps:0` are compared canonically. A requested MPS
run that produces a CPU policy is a failure, not a successful fallback.

No `PYTORCH_ENABLE_MPS_FALLBACK` or similar silent CPU-fallback environment
variable is enabled by this feature. Unsupported MPS operations must surface as
clear training failures.

## Deterministic seeding

Every CPU and MPS run uses the same explicit seed. The trainer seeds:

- Python's `random` module;
- NumPy;
- PyTorch's global generator;
- the MPS generator when MPS is selected;
- Stable-Baselines3; and
- the Gymnasium action space and vectorized environment.

These safeguards preserve the existing CPU reproducibility contract and make
each device's repeated runs as controlled as the supported libraries allow.
They do **not** imply bit-for-bit equality between CPU and MPS. Different
hardware kernels, floating-point operation order, and accelerator
implementations can produce small numerical differences even with identical
seeds and configuration.

## Canonical TRAIN-only correctness boundary

Both benchmark arms use the canonical RL data-contract loader for exactly:

```python
load_rl_partition(symbol, "train", splits_dir=...)
```

The trainer rejects a loader result that is not the TRAIN partition. Neither
the benchmark nor the trainer loads VALIDATION during gradient updates, and
TEST remains sealed and untouched.

The `rl_partition_v1` contract supplies:

- real, unscaled OHLCV execution and accounting columns;
- the configured observation features transformed by the scaler fitted on
  TRAIN only; and
- the same dynamic portfolio features used by `single_symbol_env_v1`.

The expected observation shape remains `(17,)`. The CPU and MPS runs use the
same symbol, source partition, observation features and ordering, initial
capital, transaction-cost settings, environment version, RL contract version,
seed, and PPO hyperparameters. Only the runtime device differs.

## Benchmark methodology

The safe single-symbol developer command is:

```bash
.venv/bin/python -m reinforcement_learning.training.device_benchmark \
  --symbol MCB \
  --timesteps 5120 \
  --seed 42
```

The benchmark used MCB's existing production-ready TRAIN partition and the
unchanged `ppo_single_symbol_v1` defaults:

| Setting | Controlled value |
|---|---:|
| Symbol | `MCB` |
| Seed | `42` |
| Timed requested timesteps | `5,120` |
| Warm-up requested timesteps | `512` |
| Policy | `MlpPolicy` |
| Learning rate | `3e-4` |
| Rollout steps | `512` |
| Batch size | `64` |
| Epochs per update | `10` |
| Gamma | `0.99` |
| GAE lambda | `0.95` |
| Clip range | `0.20` |
| Entropy coefficient | `0.01` |
| Value-function coefficient | `0.50` |
| Maximum gradient norm | `0.50` |

One CPU warm-up and one MPS warm-up are run before their timed counterparts so
one-time framework and accelerator initialization does not dominate the
comparison. Warm-up models remain in memory only and are discarded. The timed
CPU and MPS runs then use the same training length and configuration.

The reported duration is synchronized wall-clock time for the controlled
in-memory trainer call. MPS dispatch is asynchronous, so
`torch.mps.synchronize()` is called immediately before starting the MPS timer
and immediately after the MPS run completes. CPU timing requires no device
synchronization. A GPU duration that omits the final synchronization is invalid
and must not be compared with CPU timing.

For each timed run:

```text
timesteps_per_second = actual_timesteps / wall_clock_duration_seconds
```

The cross-device ratio is:

```text
speedup = CPU_duration_seconds / MPS_duration_seconds
```

A ratio above `1.0` means MPS was faster; below `1.0` means CPU was faster.

## Safety and data-integrity controls

The benchmark utility is deliberately single-symbol only. It has no bulk
mode, persistence option, registry writer, validation evaluator, or TEST
evaluator. It must not call `PPO.save()`, create files under production model
directories, modify `model_registry.csv`, or produce an artifact that can be
mistaken for a production model.

Before and after the final benchmark, the production registry and saved-model
directory inventories must be compared. Any change invalidates the benchmark's
safety result. Unsupported MPS operations, a device mismatch, interruption, or
another training exception must be reported as a failed run rather than hidden
as CPU execution.

Stable-Baselines3 may warn that PPO with an MLP policy can under-utilize an
accelerator and may run faster on CPU. That diagnostic is expected, is retained
in benchmark warnings, and is one reason the default is not changed merely
because MPS exists.

## Optional MPS telemetry

Telemetry is informational and never controls correctness. Where available,
the report may include:

- MPS built and available status;
- device count;
- PyTorch's MPS device name and GPU core count;
- current tensor memory allocated;
- total memory allocated by the Metal driver; and
- recommended maximum working-set memory.

PyTorch exposes these through `torch.backends.mps` and `torch.mps` on the
installed version. Each optional query is guarded independently. An unavailable
or unstable telemetry API is reported as `Not Available` and does not fail an
otherwise valid benchmark.

## Measured result

The controlled benchmark ran on 2026-08-09. Both arms completed all requested
timesteps in memory, used identical returned training provenance, and reported
the intended SB3 policy device. No model-quality conclusion is drawn.

| Runtime dependency | Installed version |
|---|---:|
| Python | 3.11.9 |
| PyTorch | 2.13.0 |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |

| Metric | CPU | MPS |
|---|---:|---:|
| Requested device | `cpu` | `mps` |
| Resolved/actual SB3 device | `cpu` / `cpu` | `mps` / `mps:0` |
| Symbol | `MCB` | `MCB` |
| TRAIN rows | 1,704 | 1,704 |
| TRAIN date range | 2016-10-06 to 2023-08-23 | 2016-10-06 to 2023-08-23 |
| Requested / actual timesteps | 5,120 / 5,120 | 5,120 / 5,120 |
| Synchronized wall-clock duration | 2.8135 seconds | 19.1594 seconds |
| Timesteps per second | 1,819.79 | 267.23 |
| Observation shape | `(17,)` | `(17,)` |
| Completion status | Completed | Completed |
| Warnings/errors | None | SB3 warned that MLP PPO may under-utilize the GPU and run slower than CPU; no error |

| Device information | Measured value |
|---|---|
| PyTorch MPS built | Yes |
| PyTorch MPS available | Yes |
| MPS device name | Apple M2 |
| MPS GPU core count | 10 |
| MPS device count | 1 |
| Current tensor allocation | 202,496 bytes (0.193 MiB) |
| Metal driver allocation | 19,529,728 bytes (18.625 MiB) |
| Recommended maximum working set | 12,713,115,648 bytes (11.840 GiB) |
| Calculated speedup (`CPU duration / MPS duration`) | **0.14685x** |

The speedup is below `1.0`, so MPS was slower. Expressed in the more intuitive
inverse form, CPU delivered approximately **6.81 times** the measured
throughput of MPS for this single-environment MLP PPO workload.

## Device decision rule

The recommendation is evidence-based and uses a deliberately conservative
materiality threshold:

- if MPS completes stably with a synchronized speedup of **at least `1.10x`**,
  recommend `mps` (or explicitly resolved `auto`) for the first PPO pilot;
- if MPS is slower, unstable, unsupported, or faster by less than `1.10x`, keep
  the production PPO default at `cpu`.

This threshold avoids changing operational defaults for timing noise or a
marginal gain. The measured MPS speedup was only `0.14685x`; therefore the
recommendation is to **keep `device="cpu"` as the PPO default and use CPU for
the first real PPO pilot on this MacBook Air M2**. Explicit MPS and AUTO support
remain available for controlled future workloads, but AUTO resolving to MPS is
not recommended for this specific pilot workload.

## Verification status

| Verification | Result |
|---|---|
| Complete `.venv/bin/python -m pytest -v` | **361 passed in 9.06 seconds; no skips; one expected SB3 MPS performance warning** |
| Hardware-gated MPS smoke test | **Passed on Apple M2; actual policy device `mps:0`** |
| `git diff --check` | **Passed** |
| `git status` reviewed | **Passed; only the ten files listed below are changed/untracked** |
| `.venv/bin/python -m pip check` | **Passed: no broken requirements** |
| TRAIN-only loader calls confirmed | **Passed; both arms used canonical `train` only** |
| VALIDATION accessed | **No** |
| TEST accessed/evaluated | **No** |
| Registry unchanged | **Yes** |
| Production model directories unchanged | **Yes** |
| Canonical training artifacts unchanged | **Yes** |
| Live HTTP requests | **None** |
| Commit | **None; automatic commit is out of scope** |

Changed files for this focused enhancement:

- `reinforcement_learning/training/devices.py`
- `reinforcement_learning/training/device_benchmark.py`
- `reinforcement_learning/training/config.py`
- `reinforcement_learning/training/ppo_trainer.py`
- `reinforcement_learning/training/results.py`
- `reinforcement_learning/training/__init__.py`
- `reinforcement_learning/model_management/persistence.py`
- `data_pipeline/tests/test_ppo_trainer.py`
- `data_pipeline/tests/test_ppo_device_benchmark.py`
- `docs/report_logs/apple_mps_ppo_benchmark.md`

## Limitations

- This is a one-symbol, one-seed, modest-length infrastructure benchmark, not a
  statistically powered performance study.
- The benchmark does not assess returns, risk-adjusted performance, model
  quality, or validation eligibility.
- CPU and MPS results are not expected to be numerically identical despite
  deterministic seeding.
- The MacBook Air is passively cooled, so thermals and other system load can
  affect wall-clock timing.
- A single MLP PPO environment often has limited accelerator utilization;
  results do not generalize automatically to larger networks, batched
  environments, or other algorithms.
- Optional telemetry availability varies by PyTorch and macOS version.
- This result does not by itself predict the exact duration of a 100,000-step
  research run, although it is sufficient for the current pilot-device choice.
