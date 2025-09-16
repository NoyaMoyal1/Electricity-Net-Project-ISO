# Train ISO — Model-Based (MBPO, PETS)

This guide covers training the ISO agent with model-based methods using `train_iso_model_based.py`. It supports MBPO and PETS, handles environment wrapping/normalization, and performs periodic evaluation and saving.

## Quickstart

```bash
# PETS (minimal working example)
python train_iso_model_based.py \
  --algo pets \
  --pcs-action-sequence pcs_actions/pattern_actions.npy \
  --iterations 2 --timesteps 480 --seed 34

# MBPO (with rollout schedule)
python train_iso_model_based.py \
  --algo mbpo \
  --pcs-action-sequence pcs_actions/pattern_actions.npy \
  --iterations 2 --timesteps 480 --seed 34 \
  --rollout-schedule 1 15 1 10
```

## Minimal Example (PETS)
```bash
python train_iso_model_based.py \
  --algo pets \
  --pcs-action-sequence pcs_actions/pattern_actions.npy \
  --iterations 1 \
  --timesteps 480 \
  --seed 0
```

## Advanced Example (PETS)
```bash
python train_iso_model_based.py \
  --algo pets \
  --pcs-action-sequence pcs_actions/pattern_actions.npy \
  --iterations 10 --timesteps 4800 --seed 34 \
  --freq-train-model 75 --num-epochs 2 \
  --pets-ensemble-size 5 --pets-horizon 15 \
  --pets-population-size 500 --pets-num-elites 50 --pets-num-iterations 5 \
  --eval-episodes 5 \
  --demand-pattern DOUBLE_PEAK --pricing-policy ONLINE --cost-type CONSTANT
```

## Flags and Arguments

- General
  - `--iterations <int>`: Outer training iterations (default: 2)
  - `--timesteps <int>`: Steps per iteration (default: 480)
  - `--seed <int>`: Random seed (default: 34)
  - `--algo {mbpo,pets}`: Model-based algorithm to use (required)
  - `--pcs-action-sequence <path.npy>`: Predefined PCS actions file (required for some flows)

- Environment
  - `--demand-pattern {CONSTANT,SINUSOIDAL,DOUBLE_PEAK,TWO_PEAK,DATA_DRIVEN}`
  - `--demand-data <path.yaml>` (required if `DATA_DRIVEN`)
  - `--pricing-policy {ONLINE,CONSTANT,QUADRATIC,INTERVALS,QUADRATIC_INTERVALS}`
  - `--cost-type {CONSTANT,VARIABLE,TIME_OF_USE}`
  - `--use-dispatch` (enable dispatch action; default True)

- Paths and IO
  - `--log-dir <dir>`: Logs directory (default: logs)
  - `--model-dir <dir>`: Models directory (default: modelsQIwithS)
  - `--plot-dir <dir>`: Plots directory (default: plotsQIwithS_copy)
  - `--eval-episodes <int>`: Episodes per evaluation (default: 5)

- MBPO-specific
  - `--rollout-schedule <min max min_epoch max_epoch>`
  - `--effective-model-rollouts-per-step <int>`
  - `--num-sac-updates-per-step <int>`
  - `--sac-updates-every-steps <int>`
  - `--freq-train-model <int>`
  - `--real-data-ratio <float>`
  - `--initial-exploration-steps <int>`
  - SAC hyperparameters: `--sac-gamma`, `--sac-tau`, `--sac-alpha`, `--sac-hidden-size`, `--sac-lr`, `--sac-target-update-interval`, `--sac-automatic-entropy-tuning`, `--sac-target-entropy`

- PETS-specific
  - Ensemble/model: `--pets-ensemble-size`, `--pets-hidden-dim`, `--pets-num-layers`, `--pets-lr`, `--pets-batch-size`, `--pets-buffer-size`
  - Planning: `--pets-horizon`, `--pets-population-size`, `--pets-num-elites`, `--pets-num-iterations`

## Troubleshooting & Tips
- DATA_DRIVEN demand requires `--demand-data`.
- Keep `norm_path` consistent across train/eval; the script manages saving to `logs/iso/vec_normalize.pkl`.
- For CUDA, set `CUDA_VISIBLE_DEVICES` appropriately. If unset, CPU is used.
- If you see shape mismatches during PETS/MBPO, verify action bounds and observation shapes printed at startup.
