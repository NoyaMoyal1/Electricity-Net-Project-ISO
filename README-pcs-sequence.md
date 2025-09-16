# PCS Sequence Generation

This guide explains how to generate predefined PCS (prosumer/battery) action sequences for use during ISO training and evaluation. Sequences can be random, policy-driven, or pattern-based and are saved to a NumPy `.npy` file for reproducibility.

## Quickstart

```bash
# Random sequence (10,000 steps)
python create_pcs_action_sequence.py \
  --output-file pcs_actions/random_actions.npy \
  --method random \
  --sequence-length 10000

# Pattern sequence (daily charge/discharge pattern, 480 steps)
python create_pcs_action_sequence.py \
  --output-file pcs_actions/pattern_actions.npy \
  --method pattern \
  --sequence-length 480 \
  --pattern-type charge_discharge_cycle \
  --cycle-length 48

# From a pre-trained policy (PPO)
python create_pcs_action_sequence.py \
  --output-file pcs_actions/policy_actions.npy \
  --method from_policy \
  --policy-path models/ppo_pcs.zip \
  --policy-type ppo \
  --demand-pattern CONSTANT --pricing-policy ONLINE --cost-type CONSTANT
```

## Flags and Arguments

- Output and length
  - `--output-file <path>`: Destination `.npy` file (directories created automatically)
  - `--sequence-length <int>`: Number of steps to generate (default: 10000)
  - `--seed <int>`: Random seed (default: 42)

- Method selection
  - `--method {random,from_policy,pattern}`: Generation method

- From policy (when `--method from_policy`)
  - `--policy-path <path>`: Path to pre-trained policy (required)
  - `--policy-type {ppo,recurrent_ppo,td3}`: Policy algorithm (default: ppo)
  - Environment parameters used to build the policy env:
    - `--demand-pattern {CONSTANT,SINUSOIDAL,DOUBLE_PEAK,TWO_PEAK,DATA_DRIVEN}`
    - `--demand-data <path.yaml>` (required if `DATA_DRIVEN`)
    - `--pricing-policy {ONLINE,CONSTANT,QUADRATIC,INTERVALS}`
    - `--cost-type {CONSTANT,VARIABLE,TIME_OF_USE}`

- Pattern generation (when `--method pattern`)
  - `--pattern-type {charge_discharge_cycle,price_responsive,yaml_config}`
  - `--cycle-length <int>`: Steps per cycle (default: 48)
  - `--pattern-config <path.yaml>`: Required when `yaml_config`

## Where outputs go

- The generated sequence is saved to the path provided by `--output-file`.
- Use this file later with training scripts via:
  - Model-based: `--pcs-action-sequence pcs_actions/pattern_actions.npy`
  - Model-free: `--pcs-action-file pcs_actions/pattern_actions.npy`

## Troubleshooting & Tips

- DATA_DRIVEN demand requires `--demand-data` when using `from_policy`.
- For recurrent policies, generation handles `lstm_states` internally.
- Ensure the policy and environment flags are consistent with how the policy was trained.
- Inspect the first few values by loading the file in Python:
  ```python
  import numpy as np
  a = np.load('pcs_actions/pattern_actions.npy')
  print(a[:10])
  ```
