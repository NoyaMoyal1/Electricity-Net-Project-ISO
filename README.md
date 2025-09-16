# Electricity-Net-Project-ISO

A cohesive toolkit for researching and engineering ISO (Independent System Operator) decision-making in a simplified electricity market. The repository provides:
- Reusable environment wrappers to expose single-agent training views over a multi-agent market
- Two complementary training families: model-based (MBPO, PETS) and model-free (PPO, RecurrentPPO, TD3)
- A utility to generate reproducible PCS action sequences used during ISO training/evaluation

This document gives you the big picture and helps you navigate to the right focused guide. It intentionally avoids duplicating CLI details and examples, which live in the per-topic READMEs.

## What you can do with this repo
- Generate PCS (prosumer/battery) action sequences for deterministic or policy-driven ISO training
- Train an ISO with model-based RL (PETS/MBPO) or model-free RL (PPO/RecurrentPPO/TD3)
- Run periodic evaluations, save models/normalization, and produce plots for analysis

## Repository map (orientation)
- `energy_net/`
  - `alternating_wrappers_model_based.py` and `alternating_wrappers_model_free.py`: wrap the multi-agent environment into ISO/PCS single-agent training interfaces, manage normalization/monitoring
  - Market logic, controllers, and environment modules under `energy_net/*`
- Top-level entry points
  - `create_pcs_action_sequence.py`: build PCS action sequences (random/pattern/policy)
  - `train_iso_model_based.py`: ISO training with PETS/MBPO
  - `train_iso_model_free.py`: ISO training with PPO/RecurrentPPO/TD3
  - `train_iso_slurm_job.sh`: example SLURM job wrapper
- Outputs (configurable)
  - `logs*/`, `models*/`, `plots*/`, `eval_plots*/`

## Typical lifecycle (high level)
1) Prepare PCS behavior (optional)
   - Generate a deterministic PCS action sequence when you want fully reproducible ISO training conditions
2) Choose a training family
   - Model-based (PETS/MBPO): learns a world model or uses model rollouts for planning
   - Model-free (PPO/RecurrentPPO/TD3): optimizes directly in the real environment interface
3) Train + evaluate
   - Training scripts create/restore normalization, log metrics, periodically evaluate, and save artifacts
4) Analyze
   - Inspect plots and logs, compare runs, and iterate with different demand patterns or pricing policies

## Configuration knobs (concepts, not flags)
- Environment scenario: demand pattern (e.g., CONSTANT, SINUSOIDAL, DATA_DRIVEN), pricing policy, cost type, dispatch on/off
- Algorithm family: model-based (PETS/MBPO) vs model-free (PPO/RecurrentPPO/TD3)
- Reproducibility: seeds, deterministic PCS sequences, and saved normalization (`logs/iso/vec_normalize.pkl`)
- Outputs: directories for models, logs, plots

## Go deeper (task-focused guides)
Use these focused READMEs for step-by-step commands, full flag documentation, and examples:
- `README-pcs-sequence.md` — Generate PCS action sequences (random/pattern/policy)
- `README-train-iso-model-based.md` — Train ISO with model-based RL (PETS/MBPO)
- `README-train-iso-model-free.md` — Train ISO with model-free RL (PPO/RecurrentPPO/TD3)

## Troubleshooting & tips (brief)
- DATA_DRIVEN scenarios require a demand YAML; the model-based/model-free guides show the exact flags
- Keep training and evaluation normalization consistent (scripts save to `logs/iso/vec_normalize.pkl`)
- GPU usage is controlled via `CUDA_VISIBLE_DEVICES` if available; otherwise CPU is used
- For clusters, use `train_iso_slurm_job.sh` as a template and adapt resources/paths
