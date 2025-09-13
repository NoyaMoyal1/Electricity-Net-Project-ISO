# Electricity-Net-Project-ISO
Project Overview

This repository contains an experimental framework for ISO-side electricity pricing and dispatch using reinforcement learning. It includes both model-free (TD3, PPO, Recurrent PPO) and model-based (PETS) pipelines, a configurable ISO/PCS simulation environment, training/evaluation scripts, and reproducible experiment setups (local and Slurm).

What this repo does

Implements an ISO environment with demand profiles, PCS actions, and pricing/dispatch signals.

Provides training scripts for TD3, PPO, Recurrent PPO, and PETS (Cross-Entropy planning over a learned dynamics model).

Includes evaluation utilities (rollouts, metrics, plots) and saved figures comparing algorithms across increasing “complexity” scenarios (cycles, peaks, added noise).

Supports local runs and cluster runs (Slurm) with job scripts and logging.
