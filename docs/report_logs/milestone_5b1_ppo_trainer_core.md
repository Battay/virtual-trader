# Milestone 5B-1: Stable-Baselines3 PPO trainer core

Implementation and verification date: **2026-08-08 (Asia/Karachi)**

## Objective

Milestone 5B-1 introduces a reusable, production-data single-symbol PPO trainer
without implementing model promotion, validation-based selection, final test
evaluation, bulk training, or Streamlit training controls. The trainer returns
the completed PPO object in memory and does not write a model or registry row.

## Dependencies and compatibility

The project requirement is pinned to `stable-baselines3==2.9.0`. PyTorch is
left as the supported transitive dependency rather than separately overriding
Stable-Baselines3's constraint.

Verified project-virtual-environment versions:

| Component | Version |
|---|---:|
| Python | 3.11.9 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.13.0 |
| Gymnasium | 1.3.0 |

Stable-Baselines3 2.9.0 supports Python 3.10 or newer, Gymnasium
`>=0.29.1,<2.0`, NumPy `>=1.20,<3.0`, and PyTorch `>=2.8,<3.0`. Installation
used only `.venv/bin/python -m pip`; `pip check` reported no broken
requirements. No optional Stable-Baselines3 extras were installed. TensorBoard
logging therefore remains disabled and is not a 5B-1 dependency.

Official references:

- [Stable-Baselines3 2.9.0 release](https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.9.0)
- [Stable-Baselines3 2.9.0 dependency metadata](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.9.0/setup.py)
- [PyTorch package metadata](https://pypi.org/project/torch/)

## Training data contract

`train_single_symbol()` makes exactly one canonical market-data request:

```python
load_rl_partition(symbol, "train", splits_dir=...)
```

It rejects a loader result whose partition is not `train`. It never requests
the validation or test partition, and it contains no evaluator or model
selection logic.

The loaded `rl_partition_v1` frame provides:

- identity and chronology through `symbol` and `date`;
- real, unscaled `open`, `high`, `low`, `close`, and `volume` for execution and
  accounting;
- the 12 configured observation features transformed by the training-fitted
  observation scaler;
- five portfolio-state features calculated dynamically by
  `single_symbol_env_v1`.

The resulting observation shape is `(17,)`. Source frames are deep-copied by
the trainer/vector-environment boundary and are never overwritten.

## Versioned PPO configuration

Configuration version: **`ppo_single_symbol_v1`**

| Parameter | Default |
|---|---:|
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
| Seed | `42` |
| Total timesteps | `100,000` |
| Device | `cpu` |

Small rollout and timestep values are accepted for deterministic tests and
developer smoke checks. Stable-Baselines3 completes full rollouts, so the
reported actual timestep count can exceed the requested count when the request
is not divisible by `n_steps`.

## Trainer architecture

- `training/config.py`: frozen configuration, version, and validation.
- `training/callbacks.py`: bounded progress events and cooperative cancellation.
- `training/results.py`: JSON-safe result metadata plus an in-memory-only model.
- `training/ppo_trainer.py`: canonical train loader, environment validation,
  deterministic one-instance `DummyVecEnv`, PPO construction, safe failure
  handling, and single-symbol CLI.

Python, NumPy, Stable-Baselines3, PyTorch, the Gymnasium action space, and the
vector environment all receive the same deterministic seed. Environment
validation uses a separate environment instance so its completed validation
episode cannot leak state into learning.

Progress is emitted at coarse intervals rather than every environment step.
Keyboard interruption, callback cancellation, and ordinary exceptions return
`interrupted` or `failed` results without an exposed model.

## Persistence and CLI safety

5B-1 never calls `PPO.save()` and never imports or invokes model-registry write
functions. The optional `output_dir` is reserved metadata for later milestones;
no artifact is written there. Paths inside `reinforcement_learning/saved_models`
or `data/models` are rejected explicitly.

The CLI is single-symbol only:

```bash
.venv/bin/python -m reinforcement_learning.training.ppo_trainer \
  --symbol MCB --timesteps 512 --seed 42 --smoke-test
```

Smoke mode is explicitly labelled and capped at 1,024 requested timesteps. A
non-smoke CLI run requires an explicit timestep count, preventing an accidental
implicit 100,000-step invocation.

## Production-data smoke verification

One integration-only run was completed. This is not a profitability or model
quality result.

| Field | Result |
|---|---|
| Symbol | MCB |
| Partition | Train only |
| Train rows | 1,704 |
| Train dates | 2016-10-06 through 2023-08-23 |
| Requested / actual timesteps | 512 / 512 |
| Seed | 42 |
| Device | CPU |
| Observation shape | `(17,)` |
| RL contract | `rl_partition_v1` |
| Environment | `single_symbol_env_v1` |
| Duration | 1.086 seconds |
| Status | Completed in memory |

No validation or test partition was loaded. No model file was saved, no model
registry entry was created, and no production model directory changed. A
before/after SHA-256 inventory of 9,745 protected raw, backfill, processed-source,
registry, and production-model files was identical.

## Limitations and next milestone

- No validation evaluator or Buy-and-Hold/Random comparison is included.
- The test partition remains sealed.
- No hyperparameter search, multi-symbol orchestration, or GPU path exists.
- The returned model is in memory only; atomic model bundle persistence and
  registry promotion remain deferred.
- TensorBoard is not enabled because its optional dependency was not forced.

Milestone **5B-2** should add validation-only policy evaluation and frozen
candidate-selection rules while continuing to keep the test partition sealed.
