# Train ISO — Model-Free (PPO, RecurrentPPO, TD3)

This guide covers training the ISO agent with model-free methods using `train_iso_model_free.py`. It supports PPO, RecurrentPPO (LSTM), and TD3. The script handles environment wrapping, normalization, saving, and periodic evaluation.

## Quickstart

```bash
# PPO (minimal working example)
python train_iso_model_free.py \
  --algorithm ppo \
  --iterations 2 --timesteps 480 --seed 34

# RecurrentPPO (LSTM) with predefined PCS sequence
python train_iso_model_free.py \
  --algorithm recurrent_ppo \
  --pcs-action-file pcs_actions/pattern_actions.npy \
  --iterations 2 --timesteps 480 --seed 34 --lstm-size 64
```

## Minimal Example (PPO)
```bash
python train_iso_model_free.py \
  --algorithm ppo \
  --iterations 1 --timesteps 480 --seed 0
```

## Advanced Example (TD3)
```bash
python train_iso_model_free.py \
  --algorithm td3 \
  --iterations 10 --timesteps 4800 --seed 123 \
  --net-arch 64 64 32 \
  --learning-rate 1e-3 --batch-size 64 \
  --buffer-size 100000 --train-freq 2 \
  --policy-noise 0.2 --final-noise 0.05 \
  --demand-pattern DOUBLE_PEAK --pricing-policy ONLINE --cost-type CONSTANT
```

## Flags and Arguments

- General
  - `--iterations <int>`: Outer training iterations (default: 20)
  - `--timesteps <int>`: Steps per iteration (default: 480)
  - `--seed <int>`: Random seed (default: 34)
  - `--algorithm {ppo,recurrent_ppo,td3}`: Algorithm (default: recurrent_ppo)

- Model-free algorithm options
  - PPO: `--net-arch <ints...>`, `--ent-coef`, `--learning-rate`, `--batch-size`
  - RecurrentPPO: `--lstm-size`, `--ent-coef`, `--learning-rate`, `--batch-size`
  - TD3: `--net-arch <ints...>`, `--learning-rate`, `--buffer-size`, `--train-freq`, `--policy-noise`, `--final-noise`

- Environment
  - `--demand-pattern {CONSTANT,SINUSOIDAL,DOUBLE_PEAK,TWO_PEAK,DATA_DRIVEN}`
  - `--demand-data <path.yaml>` (required if `DATA_DRIVEN`)
  - `--pricing-policy {ONLINE,CONSTANT,QUADRATIC,INTERVALS,QUADRATIC_INTERVALS}`
  - `--cost-type {CONSTANT,VARIABLE,TIME_OF_USE}`
  - `--use-dispatch` (enable dispatch action; default True)

- PCS inputs
  - `--pcs-action-file <path.npy>`: Sequence to drive PCS during ISO training (optional)
  - `--pcs-model <path.zip>`: Use a trained PCS model (optional)
  - `--pcs-norm-path <path.pkl>`: Normalization for PCS model env (optional)

- Paths and IO
  - `--log-dir <dir>`: Logs directory (default: logs)
  - `--model-dir <dir>`: Models directory (default: modelsQIwithS)
  - `--plot-dir <dir>`: Plots directory (default: plotsQIwithS)
  - `--eval-episodes <int>`: Episodes per evaluation (default: 5)
  - Eval-only mode: `--eval-only`, `--best-model <path>`, `--norm-path <path>`

## Troubleshooting & Tips
- DATA_DRIVEN demand requires `--demand-data`.
- Normalization is saved to `logs/iso/vec_normalize.pkl`; keep it consistent for evaluation.
- For recurrent policies, the script manages `lstm_states` and `episode_start` flags.
- If action scaling looks off, confirm wrappers and `RescaleAction` are applied (see `energy_net/alternating_wrappers_model_free.py`).
