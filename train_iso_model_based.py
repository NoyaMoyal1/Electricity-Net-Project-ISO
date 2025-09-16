import argparse
import numpy as np
import json
from energy_net.alternating_wrappers_model_based import make_iso_env
import os
from plot_callback import PlotCallback
from omegaconf import OmegaConf, DictConfig
# from training_monitor import TrainingMonitor, ActionProbe  # Monitoring disabled for speed

# Suppress Gymnasium deprecation warnings
import warnings
import gymnasium as gym

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", message=r".*env\..*to get variables from other wrappers is deprecated.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained since 2022.*")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
# --- Argument parsing (as before) ---
def parse_args():
    """Parse command line arguments (copied from train_iso_model_free.py)"""
    parser = argparse.ArgumentParser(description="Train ISO agent with model-based RL (MBPO)")
    # Training parameters
    parser.add_argument("--iterations", type=int, default=2, 
                        help="Number of training iterations")
    parser.add_argument("--timesteps", type=int, default=480, 
                        help="Steps per iteration (480 = 10 days)")
    parser.add_argument("--seed", type=int, default=34, 
                        help="Random seed")
    
    # Environment parameters
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern type")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE", 
                        choices=["ONLINE", "CONSTANT", "QUADRATIC", "INTERVALS", "QUADRATIC_INTERVALS"],
                        help="Pricing policy type")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type")
    parser.add_argument("--use-dispatch", action="store_false", dest="use_dispatch", default=True,
                        help="Disable dispatch action for ISO (default: enabled)")
    
    # PCS behavior control
    parser.add_argument("--pcs-action-sequence", type=str, required=True,
                        help="Path to a numpy file (.npy) containing a sequence of predefined PCS actions")
    # Algorithm parameters
    parser.add_argument("--algo", type=str, required=True, choices=["mbpo", "pets"], help="Algorithm to use: mbpo or pets")
    
    # MBPO-specific parameters
    parser.add_argument("--rollout-schedule", nargs="+", type=int, default=[1, 15, 1, 1],
                        help="MBPO rollout schedule [min, max, min_epoch, max_epoch]")
    parser.add_argument("--effective-model-rollouts-per-step", type=int, default=10,
                        help="MBPO effective model rollouts per step")
    parser.add_argument("--num-sac-updates-per-step", type=int, default=20,
                        help="MBPO number of SAC updates per step")
    parser.add_argument("--sac-updates-every-steps", type=int, default=1,
                        help="MBPO SAC updates every N steps")
    parser.add_argument("--freq-train-model", type=int, default=10,
                        help="MBPO frequency to train model")
    parser.add_argument("--real-data-ratio", type=float, default=0.0,
                        help="MBPO real data ratio")
    parser.add_argument("--initial-exploration-steps", type=int, default=20,
                        help="MBPO initial exploration steps")
    
    # SAC agent parameters (for MBPO)
    parser.add_argument("--sac-gamma", type=float, default=0.99,
                        help="SAC discount factor")
    parser.add_argument("--sac-tau", type=float, default=0.005,
                        help="SAC target network update rate")
    parser.add_argument("--sac-alpha", type=float, default=0.2,
                        help="SAC entropy coefficient")
    parser.add_argument("--sac-hidden-size", type=int, default=256,
                        help="SAC hidden layer size")
    parser.add_argument("--sac-lr", type=float, default=0.0003,
                        help="SAC learning rate")
    parser.add_argument("--sac-target-update-interval", type=int, default=4,
                        help="SAC target update interval")
    parser.add_argument("--sac-automatic-entropy-tuning", action="store_true", default=True,
                        help="SAC automatic entropy tuning")
    parser.add_argument("--sac-target-entropy", type=float, default=-0.2,
                        help="SAC target entropy")

    # (PlaNet parameters removed)

    # Model parameters (for PlaNet)
    parser.add_argument("--sequence-length", type=int, default=30,
        help="Length of sequence windows sampled from experience for training the model. "
            "Affects the model's ability to capture long-term dependencies.")

    parser.add_argument("--batch-size", type=int, default=15,
        help="Number of sequences per training batch.")

    parser.add_argument("--num-grad-updates", type=int, default=20,
        help="Number of gradient descent updates per model training step.")

    parser.add_argument("--free-nats", type=float, default=3.0,
        help="Threshold below which KL divergence is not penalized. "
            "Prevents over-regularization during training.")

    parser.add_argument("--kl-scale", type=float, default=1.0,
        help="Weight of the KL divergence term in the model loss. "
            "Higher values enforce stronger adherence to prior distribution.")

    # PETS-specific parameters
    parser.add_argument("--pets-ensemble-size", type=int, default=5,
        help="Number of models in the PETS ensemble")
    parser.add_argument("--pets-horizon", type=int, default=15,
        help="Planning horizon for PETS")
    parser.add_argument("--pets-population-size", type=int, default=500,
        help="Population size for CEM planner in PETS")
    parser.add_argument("--pets-num-elites", type=int, default=50,
        help="Number of elite samples for CEM planner in PETS")
    parser.add_argument("--pets-num-iterations", type=int, default=5,
        help="Number of CEM iterations in PETS")
    parser.add_argument("--pets-hidden-dim", type=int, default=200,
        help="Hidden dimension for PETS dynamics models")
    parser.add_argument("--pets-num-layers", type=int, default=4,
        help="Number of layers in PETS dynamics models")
    parser.add_argument("--pets-lr", type=float, default=1e-3,
        help="Learning rate for PETS dynamics models")
    parser.add_argument("--pets-batch-size", type=int, default=256,
        help="Batch size for PETS training")
    parser.add_argument("--pets-buffer-size", type=int, default=100000,
        help="Buffer size for PETS experience replay")

    # Paths
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="modelsQIwithS",
                        help="Directory for saved models")
    parser.add_argument("--plot-dir", type=str, default="plotsQIwithS_copy",
                        help="Directory for plots")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")

    # Continue training from existing models
    parser.add_argument("--continue-from-best", action="store_true",
                        help="Continue training from the best ISO model saved previously")
    parser.add_argument("--initial-iso-model", type=str, default=None,
                        help="Path to initial ISO model for continued training")
    parser.add_argument("--pcs-model", type=str, default=None,
                        help="Path to PCS model to use during ISO training (optional)")
    parser.add_argument("--pcs-norm-path", type=str, default=None,
                        help="Path to VecNormalize stats for PCS model normalization (optional)")
    parser.add_argument("--start-iteration", type=int, default=1,
                        help="Starting iteration number when continuing training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate the best ISO model and exit")
    parser.add_argument("--best-model", type=str, default=None,
                        help="Path to the best ISO model for evaluation")
    parser.add_argument("--norm-path", type=str, default=None,
                        help="Path to VecNormalize stats file for evaluation")
    
    # Generalization for model-based RL
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--results-path", type=str, default="training_log.json")
    parser.add_argument("--model-path", type=str, default="final_model.pt")

    return parser.parse_args()

# --- Config builders for MBPO (mbrl 0.2.0) ---
def build_mbpo_cfg(args, env):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Complete SAC agent config with environment dimensions
    obs_dim = obs_shape[0] if len(obs_shape) > 0 else 1
    act_dim = act_shape[0] if len(act_shape) > 0 else 1
    
    # Get action space bounds
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
    else:
        # Default bounds if not available
        action_low = -np.ones(act_dim)
        action_high = np.ones(act_dim)

    cfg = {
        "log_frequency_agent": 10,
        "seed": args.seed,
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "debug_mode": False,
        "overrides": {
            "normalize": True,
            "model_batch_size": 60,  # Increased from 33 for better gradient estimates
            "validation_ratio": 0.1,  # Increased from 0.05 for better validation
            "epoch_length": args.timesteps,
            "effective_model_rollouts_per_step": getattr(args, "effective_model_rollouts_per_step", 10),
            "model_lr": 5e-4,  # Reduced from 1e-3 for more stable training
            "model_wd": 1e-4,  # Increased from 1e-5 for better regularization
            "freq_train_model": getattr(args, "freq_train_model", 50),
            "num_steps": 50,
            "rollout_schedule": [1, 15, 1, 10],  
            "num_epochs_to_retain_sac_buffer": 20,  # Increased from 10 for more experience retention
            "num_sac_updates_per_step": 40,  # Increased from 20 for more frequent policy updates
            "sac_updates_every_steps": 2,  # Increased from 1 for more stable training
            "sac_batch_size": 256,  # Increased from 128 for better gradient estimates
        },
        "algorithm": {
            "num_eval_episodes": 5,
            "initial_exploration_steps": getattr(args, "initial_exploration_steps", 50),  # Increased from 20
            "random_initial_explore": True,  # Changed from False for better initial exploration
            "normalize_double_precision": False,
            "learned_rewards": True,  # Keep as True to avoid None reward issues
            "dataset_size": 500000,  # Reduced from 1000000 for faster training
            "trial_length": 480,
            "freq_train_model": getattr(args, "freq_train_model", 50),
            "real_data_ratio": 0.2,
            "target_is_delta": True,
            "normalize": True,
            # Agent hydra instantiation config uses local factory to satisfy positional signature
            "agent": {
                "_target_": "energy_net.mbrl_adapters.make_sac",
                "num_inputs": obs_dim,  # Completed from env.observation_space
                "action_space": {      # Completed from env.action_space
                    "low": action_low.tolist(),
                    "high": action_high.tolist(),
                    "shape": list(act_shape)
                },
                "args": {
                    "gamma": getattr(args, 'sac_gamma', 0.99),
                    "tau": getattr(args, 'sac_tau', 0.005),
                    "alpha": getattr(args, 'sac_alpha', 0.2),
                    "actor_hidden_dim": getattr(args, 'sac_hidden_size', 256),
                    "critic_hidden_dim": getattr(args, 'sac_hidden_size', 256),
                    "lr": getattr(args, 'sac_lr', 3e-4),
                    "target_update_interval": getattr(args, 'sac_target_update_interval', 4),
                    "automatic_entropy_tuning": getattr(args, 'sac_automatic_entropy_tuning', True),
                    "target_entropy": getattr(args, 'sac_target_entropy', -0.5)
                }
            },
        },
        # Minimal dynamics model config; create_one_dim_tr_model will fill defaults
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "ensemble_size": 5,  # Increased from 5 for better uncertainty estimation
            "propagation_method": "expectation",
            "device": "${device}"
        },
        # Replay buffer defaults used by create_replay_buffer
        "replay_buffer": {
            "capacity": 100000,
        },
    }
    return OmegaConf.create(cfg)


# (PlaNet config removed)


def build_pets_cfg(args, env):
    """Build PETS configuration similar to MBPO/PlaNet config functions."""
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    # Get environment dimensions
    obs_dim = int(obs_shape[0]) if len(obs_shape) > 0 else 1
    act_dim = int(act_shape[0]) if len(act_shape) > 0 else 1
    
    # Get action space bounds
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low.astype(float)
        action_high = env.action_space.high.astype(float)
    else:
        # Default bounds if not available
        action_low = -np.ones(act_dim, dtype=float)
        action_high = np.ones(act_dim, dtype=float)
    
    cfg = {
        "seed": args.seed,
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "debug_mode": False,
        "overrides": {
            "normalize": True,
            "model_batch_size": getattr(args, "pets_batch_size", 256),
            "validation_ratio": 0.1,
            "num_epochs_train_model": 5,  # Required by train_model_and_save_model_and_data
            "patience": 10,  # Required by train_model_and_save_model_and_data
            "bootstrap_permutes": False,  # Required by train_model_and_save_model_and_data
            "epoch_length": args.timesteps,
            "model_lr": getattr(args, "pets_lr", 1e-3),
            "model_wd": 1e-4,
            "freq_train_model": getattr(args, "freq_train_model", 50),
            "num_steps": 50,
            "learned_rewards": True,  # Use learned rewards like MBPO
            "target_is_delta": True,
        },
        "algorithm": {
            "num_eval_episodes": 5,
            "initial_exploration_steps": getattr(args, "initial_exploration_steps", 50),
            "random_initial_explore": True,
            "normalize_double_precision": False,
            "learned_rewards": True,
            "dataset_size": getattr(args, "pets_buffer_size", 100000),
            "trial_length": args.timesteps,
            "freq_train_model": getattr(args, "freq_train_model", 50),
            "target_is_delta": True,
            "normalize": True,
            # CEM-based planning agent
            "agent": {
                "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
                "planning_horizon": getattr(args, "pets_horizon", 15),
                "optimizer_cfg": {
                    "_target_": "mbrl.planning.CEMOptimizer",
                    "num_iterations": getattr(args, "pets_num_iterations", 5),
                    "elite_ratio": 0.1,  # Fixed float value
                    "population_size": getattr(args, "pets_population_size", 500),
                    "alpha": 0.1,
                    "clipped_normal": True,
                    "device": "${device}",  # Reference to the device from the main config
                },
                "action_lb": [float(x) for x in action_low.tolist()],
                "action_ub": [float(x) for x in action_high.tolist()],
            },
        },
        # GaussianMLP ensemble for dynamics model
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "ensemble_size": getattr(args, "pets_ensemble_size", 5),
            "propagation_method": "expectation",
            "device": "${device}",
        },
        # Replay buffer for PETS
        "replay_buffer": {
            "capacity": getattr(args, "pets_buffer_size", 100000),
        },
    }
    return OmegaConf.create(cfg)

# --- Adapter wrappers ---
def _rebuild_sac_from_checkpoint(sac_path, env, cfg):
    """
    Rebuild SAC agent from checkpoint with proper environment dimensions.
    
    Args:
        sac_path: Path to saved SAC checkpoint
        env: Environment to get dimensions from
        cfg: Configuration used for training
        
    Returns:
        Rebuilt SAC agent with loaded weights
    """
    import torch
    from energy_net.mbrl_adapters import make_sac
    
    # Load the saved checkpoint
    checkpoint = torch.load(sac_path, map_location='cpu')
    
    # Get environment dimensions
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    obs_dim = obs_shape[0] if len(obs_shape) > 0 else 1
    act_dim = act_shape[0] if len(act_shape) > 0 else 1
    
    # Get action space bounds
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
    else:
        action_low = -np.ones(act_dim)
        action_high = np.ones(act_dim)
    
    # Create action space dict
    action_space = {
        "low": action_low.tolist(),
        "high": action_high.tolist(),
        "shape": list(act_shape)
    }
    
    # Get SAC args from config
    sac_args = cfg.algorithm.agent.args
    
    # Rebuild SAC agent
    sac_agent = make_sac(obs_dim, action_space, sac_args)
    
    # Load weights from checkpoint
    if 'actor' in checkpoint:
        sac_agent.actor.load_state_dict(checkpoint['actor'])
    if 'critic' in checkpoint:
        sac_agent.critic.load_state_dict(checkpoint['critic'])
    if 'critic_target' in checkpoint:
        sac_agent.critic_target.load_state_dict(checkpoint['critic_target'])
    if 'log_alpha' in checkpoint:
        sac_agent.log_alpha.data = checkpoint['log_alpha']
    
    return sac_agent

def _policy_action(agent, obs):
    """
    Unified action selection that tries different agent APIs in order.
    
    Args:
        agent: The agent to get action from
        obs: Observation to act on
        
    Returns:
        Action from the agent
        
    Raises:
        RuntimeError: If no compatible action method is found
    """
    import numpy as np
    # Try predict method first (most common)
    if hasattr(agent, 'predict'):
        try:
            action = agent.predict(obs, deterministic=True)
            # Handle tuple return (action, state) or just action
            if isinstance(action, tuple):
                action = action[0]
            return action
        except Exception as e:
            pass
    
    # Try act method
    if hasattr(agent, 'act'):
        try:
            # Ensure observation is properly formatted for TrajectoryOptimizerAgent
            if isinstance(obs, tuple):
                obs = obs[0]  # Extract observation from tuple
            if isinstance(obs, np.ndarray):
                obs = obs.astype(np.float32)  # Ensure correct dtype
                if obs.ndim == 0:  # Scalar
                    obs = np.array([obs], dtype=np.float32)
                elif obs.ndim > 1:  # Flatten if needed
                    obs = obs.flatten()
            action = agent.act(obs)
            return action
        except Exception as e:
            print(f"act method failed: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            pass
    
    # Try select_action method (without eval parameter first)
    if hasattr(agent, 'select_action'):
        try:
            action = agent.select_action(obs)
            return action
        except Exception as e:
            # Try with eval parameter if the first attempt failed
            try:
                action = agent.select_action(obs, eval=True)
                return action
            except Exception as e2:
                pass
    
    # Try direct actor call as fallback
    if hasattr(agent, 'actor'):
        try:
            import torch
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if hasattr(agent, 'device'):
                obs_tensor = obs_tensor.to(agent.device)
            with torch.no_grad():
                action = agent.actor(obs_tensor)
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                return action
        except Exception as e:
            pass
    
    # If we get here, no compatible method found
    raise RuntimeError(f"Agent does not have a compatible action method. Available methods: {[m for m in dir(agent) if not m.startswith('_')]}")

class MBPOAdapter:
    def __init__(self, env, test_env, term_fn, cfg, work_dir):
        self.env = env
        self.test_env = test_env
        self.term_fn = term_fn
        self.cfg = cfg
        self.work_dir = work_dir
        self.agent = None  # Will store the trained agent after training
        self.monitor = None  # Will be set during training
        
    def train(self, num_epochs=None):
        import mbrl.algorithms.mbpo as mbpo
        
        # Simple MBPO training without plotting wrapper
        results = mbpo.train(self.env, self.test_env, self.term_fn, self.cfg, silent=False, work_dir=self.work_dir)
        
        # Store the trained agent for evaluation
        if hasattr(results, 'agent'):
            self.agent = results.agent
        elif hasattr(results, 'trainer') and hasattr(results.trainer, 'agent'):
            self.agent = results.trainer.agent
        else:
            # Try to load the agent from the saved model in work_dir
            try:
                sac_path = os.path.join(self.work_dir, "sac.pth")
                if os.path.exists(sac_path):
                    # Rebuild SAC agent from checkpoint
                    self.agent = _rebuild_sac_from_checkpoint(sac_path, self.env, self.cfg)
            except Exception as e:
                print(f"Warning: Failed to load agent from checkpoint: {e}")
        
        # Monitoring finished (disabled)
        
        return results
        
    def save(self, path):
        # mbpo.train handles logging/checkpointing internally via work_dir; nothing explicit to save here
        pass

# (PlaNet adapter removed)

class PETSAdapter:
    """PETS (Cross-Entropy MPC) adapter using mbrl-lib components."""
    
    def __init__(self, env, cfg, work_dir):
        self.env = env
        self.cfg = cfg
        self.work_dir = work_dir
        self.agent = None  # Will store the trained agent after training
        self.monitor = None  # Will be set during training
        
    def train(self, num_epochs=None):
        """Train PETS using the correct mbrl-lib modular approach."""
        try:
            import mbrl.util.common as common_util
            import mbrl.models as models
            import mbrl.planning as planning
            import mbrl.env.termination_fns as termination_fns
            import torch
            import numpy as np
            
            print("Initializing PETS with correct mbrl-lib implementation...")
            
            # Get environment dimensions
            obs_shape = self.env.observation_space.shape
            act_shape = self.env.action_space.shape
            
            # Convert to numpy arrays if they are tuples, then to Python ints for OmegaConf
            if isinstance(obs_shape, tuple):
                obs_shape = np.array(obs_shape)
            if isinstance(act_shape, tuple):
                act_shape = np.array(act_shape)
            
            # Convert numpy int64 to Python int for OmegaConf compatibility
            obs_shape = [int(x) for x in obs_shape]
            act_shape = [int(x) for x in act_shape]
            
            # Create termination function (no termination for continuous tasks)
            term_fn = termination_fns.no_termination
            
            # Create dynamics model using the configuration
            dynamics_model = common_util.create_one_dim_tr_model(
                self.cfg, obs_shape, act_shape
            )
            
            # Create replay buffer
            rng = np.random.default_rng(seed=self.cfg.seed)
            replay_buffer = common_util.create_replay_buffer(
                self.cfg, obs_shape, act_shape, rng=rng
            )
            
            # Create model environment
            device = self.cfg.device
            generator = torch.Generator(device=device)
            model_env = models.ModelEnv(
                self.env, dynamics_model, term_fn, generator=generator
            )
            
            # Create planning agent (TrajectoryOptimizerAgent with CEM)
            agent_cfg = self.cfg.algorithm.agent
            agent = planning.create_trajectory_optim_agent_for_model(
                model_env, agent_cfg, num_particles=15
            )
            
            # Create model trainer
            model_trainer = models.ModelTrainer(
                dynamics_model, 
                optim_lr=self.cfg.overrides.model_lr, 
                weight_decay=self.cfg.overrides.model_wd
            )
            
            # Training loop
            num_trials = num_epochs or 1
            trial_length = self.cfg.overrides.epoch_length
            
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")
                
                # Reset environment and agent
                obs, _ = self.env.reset()  # Extract observation from (obs, info) tuple
                agent.reset()
                
                # Collect data for this trial
                for step in range(trial_length):
                    # Get action from agent
                    action = agent.act(obs)
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Store transition in replay buffer
                    replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
                    
                    obs = next_obs
                    
                    if done:
                        obs, _ = self.env.reset()  # Extract observation from (obs, info) tuple
                        agent.reset()
                
                # Train dynamics model periodically
                if trial % self.cfg.overrides.freq_train_model == 0:
                    print(f"Training dynamics model at trial {trial}")
                    
                    # Train the model using the proper mbrl-lib function
                    if len(replay_buffer) > 0:
                        # Use the lower-level approach with explicit parameter values
                        # Get training and validation iterators from replay buffer
                        train_iter, val_iter = common_util.get_basic_buffer_iterators(
                            replay_buffer,
                            batch_size=256,  # Explicit value
                            val_ratio=0.1,   # Explicit value
                            ensemble_size=3, # Explicit value
                            shuffle_each_epoch=True,
                            bootstrap_permutes=False
                        )
                        
                        # Debug: Check the first batch to see data dimensions
                        try:
                            first_batch = next(iter(train_iter))
                            print(f"[DEBUG] First batch type: {type(first_batch)}")
                            print(f"[DEBUG] First batch attributes: {dir(first_batch)}")
                            if hasattr(first_batch, 'obs'):
                                print(f"[DEBUG] Batch obs shape: {first_batch.obs.shape}")
                            if hasattr(first_batch, 'action'):
                                print(f"[DEBUG] Batch action shape: {first_batch.action.shape}")
                            if hasattr(first_batch, 'next_obs'):
                                print(f"[DEBUG] Batch next_obs shape: {first_batch.next_obs.shape}")
                            if hasattr(first_batch, 'reward'):
                                print(f"[DEBUG] Batch reward shape: {first_batch.reward.shape}")
                            if hasattr(first_batch, 'actions'):
                                print(f"[DEBUG] Batch actions shape: {first_batch.actions.shape}")
                            if hasattr(first_batch, 'rewards'):
                                print(f"[DEBUG] Batch rewards shape: {first_batch.rewards.shape}")
                            if hasattr(first_batch, 'act'):
                                print(f"[DEBUG] Batch act shape: {first_batch.act.shape}")
                        except Exception as e:
                            print(f"[DEBUG] Could not inspect batch: {e}")
                        
                        # Train the model using the iterators
                        model_trainer.train(
                            train_iter, 
                            val_iter,
                            num_epochs=5,  # Explicit value
                            patience=10    # Explicit value
                        )
                        
                        # Update model normalizer
                        dynamics_model.update_normalizer(replay_buffer.get_all())
            
            # Store the trained agent
            self.agent = agent
            
            print("PETS training completed successfully!")
            return {"agent": agent, "dynamics_model": dynamics_model}
            
        except Exception as e:
            error_msg = f"PETS training failed with error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
        
    def save(self, path):
        # pets.train handles logging/checkpointing internally via work_dir; nothing explicit to save here
        pass

# Utility termination function: no termination inside model env by default
def no_termination_fn(states, actions):
    import torch
    return torch.zeros((states.shape[0], 1), dtype=torch.bool, device=states.device)

# --- Model creation functions ---
def create_mbpo_trainer(env, args, **kwargs):
    import mbrl.algorithms.mbpo as mbpo
    import mbrl.util.common as mbrl_common
    cfg = build_mbpo_cfg(args, env)
    # Runtime verification of learned rewards and model output size
    try:
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        dynamics_model_tmp = mbrl_common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        print("[DBG] learned_rewards =", getattr(dynamics_model_tmp, "learned_rewards", None))
        try:
            print("[DBG] model_out_size =", getattr(dynamics_model_tmp.model, "out_size", None))
        except Exception:
            pass
        del dynamics_model_tmp
    except Exception as _e:
        print(f"[DBG] Skipped model cfg verification due to: {_e}")
    # test/eval env separate
    eval_env = make_env(
        args.pcs_action_sequence,
        steps_per_iteration=args.timesteps,
        cost_type=args.cost_type,
        pricing_policy=args.pricing_policy,
        demand_pattern=args.demand_pattern,
        seed=args.seed + 777,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        use_dispatch_action=args.use_dispatch,
        demand_data_path=args.demand_data,
        eval_mode=True,
        norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"
    )
    # Wrap environments to ensure reset() returns (obs, info) and step() returns 5-tuple for MBPO compatibility
    class ResetInfoAdapter:
        def __init__(self, env):
            self.env = env
        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            # Ensure we return (obs, info) format
            if isinstance(out, tuple) and len(out) == 2:
                obs, info = out
            else:
                obs, info = out, {}
            
            # Ensure observation is numpy array
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            
            return obs, info
            
        def step(self, action):
            # Ensure action is in correct format
            if isinstance(action, np.ndarray):
                # Ensure action is 1D if environment expects it
                if hasattr(self.env, 'action_space') and len(self.env.action_space.shape) == 1:
                    action = action.flatten()
            
            out = self.env.step(action)
            
            # Convert to 5-tuple format that MBRL expects
            if len(out) == 4:  # (obs, reward, done, info) -> (obs, reward, terminated, truncated, info)
                obs, reward, done, info = out
                # Ensure observation is numpy array
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)
                # Ensure reward is scalar
                if isinstance(reward, (list, np.ndarray)):
                    reward = float(reward[0]) if len(reward) > 0 else 0.0
                else:
                    reward = float(reward)
                return obs, reward, bool(done), False, info
            elif len(out) == 5:  # Already 5-tuple
                obs, reward, terminated, truncated, info = out
                # Ensure observation is numpy array
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)
                # Ensure reward is scalar
                if isinstance(reward, (list, np.ndarray)):
                    reward = float(reward[0]) if len(reward) > 0 else 0.0
                else:
                    reward = float(reward)
                return obs, reward, bool(terminated), bool(truncated), info
            else:
                raise ValueError(f"Unexpected step return format: {len(out)} elements")
                
        def __getattr__(self, name):
            if name == 'seed':
                # Handle seeding gracefully if the underlying env doesn't support it
                def safe_seed(seed_val):
                    try:
                        return self.env.seed(seed_val)
                    except AttributeError:
                        # Environment doesn't support seeding, just return None
                        return None
                return safe_seed
            return getattr(self.env, name)
    
    # Wrap both training and test environments
    train_env = ResetInfoAdapter(env)
    test_env = ResetInfoAdapter(eval_env)
    
    work_dir = os.path.join(args.log_dir, "mbpo")
    os.makedirs(work_dir, exist_ok=True)
    return MBPOAdapter(train_env, test_env, no_termination_fn, cfg, work_dir)


# (PlaNet trainer factory removed)

def create_pets_trainer(env, args, **kwargs):
    """Create PETS trainer using the PETSAdapter."""
    cfg = build_pets_cfg(args, env)
    work_dir = os.path.join(args.log_dir, "pets")
    os.makedirs(work_dir, exist_ok=True)
    return PETSAdapter(env, cfg, work_dir)

def make_env(pcs_action_sequence_path, **env_kwargs):
    pcs_action_sequence = np.load(pcs_action_sequence_path)
    env = make_iso_env(
        pcs_action_sequence=pcs_action_sequence,
        **env_kwargs
    )
    return env

def get_trainer(algo, env, args):
    algo = algo.lower()
    if algo == "mbpo":
        return create_mbpo_trainer(env, args)
    # (PlaNet branch removed)
    elif algo == "pets":
        return create_pets_trainer(env, args)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def evaluate_iso(agent, env_config, override_env=None, pcs_action_sequence=None, num_episodes=5, seed=None, plot_dir=None, monitor=None):
    """
    Evaluate ISO agent performance by running episodes and plotting results.
    Args:
        agent: Trained ISO agent (MBPO, etc.)
        env_config: Environment configuration dict
        override_env: Optional pre-configured environment to use
        pcs_action_sequence: Optional sequence of predefined PCS actions to use
        num_episodes: Number of episodes to run
        seed: Random seed
        plot_dir: Directory to save plots
        monitor: Optional training monitor for environment monitoring
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating ISO agent for {num_episodes} episodes...")



    # Use provided wrapped environment if available, else create properly wrapped env
    if override_env is not None:
        eval_env = override_env
        if hasattr(eval_env, 'training'):
            eval_env.training = False
        if hasattr(eval_env, 'norm_reward'):
            eval_env.norm_reward = False
    else:
        from energy_net.alternating_wrappers_model_based import make_iso_env
        eval_env = make_iso_env(
            steps_per_iteration=env_config.get('timesteps', 480),
            cost_type=env_config.get('cost_type', 'CONSTANT'),
            pricing_policy=env_config.get('pricing_policy', 'ONLINE'),
            demand_pattern=env_config.get('demand_pattern', 'CONSTANT'),
            log_dir=env_config.get('log_dir', 'logsQIwithS'),
            model_dir=env_config.get('model_dir', 'modelsQIwithS'),
            plot_dir=env_config.get('plot_dir', 'plotsQIwithS_copy'),
            pcs_policy=None,
            pcs_action_sequence=pcs_action_sequence,
            use_dispatch_action=env_config.get('use_dispatch', True),
            norm_path=env_config.get('norm_path', None),
            eval_mode=True,
            demand_data_path=env_config.get('demand_data', None)
        )

    total_iso_reward = 0
    episode_metrics = []
    eval_plots_dir = plot_dir or os.path.join("eval_plotsQIwithS_copy", "iso")
    os.makedirs(eval_plots_dir, exist_ok=True)
    # Initialize a single plotting callback for all evaluation episodes
    eval_callback = PlotCallback(verbose=0)
    eval_callback.agent_name = "iso_eval"
    eval_callback.save_path = eval_plots_dir
    eval_callback.all_episodes_actions = []

    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode+1}/{num_episodes}")
        if seed is not None:
            eval_env.seed(seed + episode)
        obs = eval_env.reset()[0]
        done = False
        episode_iso_reward = 0
        episode_steps = 0
        episode_data = []
        while not done:
            # Use unified action selection that tries different agent APIs
            try:
                action = _policy_action(agent, obs)
            except Exception as e:
                print(f"[ERROR] Failed to get action from agent: {e}")
                raise RuntimeError(f"Agent does not have a compatible action method for evaluation: {e}")
            if isinstance(action, np.ndarray) and action.ndim == 1:
                batch_action = action[np.newaxis, :]
            else:
                batch_action = np.array([action])
            step_result = eval_env.step(batch_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            done = done.any() if hasattr(done, 'any') else done
            iso_reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            episode_iso_reward += iso_reward
            episode_steps += 1
            step_info = info[0] if isinstance(info, list) else info
            step_data = {
                'step': episode_steps,
                'action': float(action[0]) if isinstance(action, np.ndarray) and action.ndim == 1 else (float(action[0, 0]) if isinstance(action, np.ndarray) and action.ndim == 2 else float(action)),
                'iso_action': action,
                'pcs_action': step_info.get('pcs_action', np.zeros(1)),
                'predicted_demand': step_info.get('predicted_demand', 0),
                'realized_demand': step_info.get('realized_demand', 0),
                'battery_level': step_info.get('battery_level', 0),
                'iso_sell_price': step_info.get('iso_sell_price', 0),
                'iso_buy_price': step_info.get('iso_buy_price', 0),
                'net_exchange': step_info.get('net_exchange', 0),
                'dispatch': step_info.get('dispatch', 0),
                'dispatch_cost': step_info.get('dispatch_cost', 0),
                'reserve_cost': step_info.get('reserve_cost', 0),
                'shortfall': step_info.get('shortfall', 0),
                'pcs_exchange_cost': step_info.get('pcs_exchange_cost', 0),
                'iso_reward': iso_reward,
            }
            for key, value in step_info.items():
                if isinstance(key, str) and key.startswith('background_'):
                    step_data[key] = value
            episode_data.append(step_data)
            
            # Record step for environment monitoring
            if monitor is not None:
                monitor.after_training_step(eval_env, obs, reward, done, info)
        

                
        total_iso_reward += episode_iso_reward
        episode_metrics.append({
            'episode': episode,
            'iso_reward': episode_iso_reward,
            'steps': episode_steps,
            'data': episode_data
        })
        # Accumulate and plot this episode using the shared callback (avoid overwriting)
        eval_callback.all_episodes_actions.append(episode_data)
        eval_callback.plot_episode_results(episode, eval_plots_dir)
        print(f"Episode {episode+1} complete: ISO reward={episode_iso_reward:.2f}")

    avg_iso_reward = total_iso_reward / num_episodes
    print(f"\nEvaluation complete:")
    print(f"Average ISO reward: {avg_iso_reward:.2f}")
    print(f"Plots saved to: {eval_plots_dir}")

    # Run random baseline for comparison if monitor is available
    if monitor is not None:
        monitor.env_monitor.run_random_baseline(eval_env, num_episodes=3)

    # Optionally, add statistics and file writing as in the original function
    return {
        'avg_iso_reward': avg_iso_reward,
        'episodes': episode_metrics,

    }


def main():
    # Suppress PCS action clipping warnings
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*Action at index.*is outside the PCS action space.*")
    logging.getLogger("alternating_wrappers_model_based").setLevel(logging.ERROR)
    
    # Patch MBRL logger to handle numpy arrays properly
    import mbrl.util.logger as mbrl_logger
    import numpy as np
    
    # Store the original _format method
    original_format = mbrl_logger.MetersGroup._format
    
    def patched_format(self, key, value, ty):
        """Patched _format method that handles numpy arrays properly"""
        if isinstance(value, np.ndarray):
            if value.size == 1:
                # Single element array, convert to scalar
                value = float(value.item())
            else:
                # Multi-element array, convert to string representation
                value = str(value)
        return original_format(key, value, ty)
    
    # Apply the patch
    mbrl_logger.MetersGroup._format = patched_format
    
    args = parse_args()

    # Check if DATA_DRIVEN pattern requires a data file
    if args.demand_pattern == "DATA_DRIVEN" and not args.demand_data:
        print("ERROR: DATA_DRIVEN demand pattern requires a demand data file.")
        print("Please specify the file path using --demand-data")
        return

    # Create all necessary directories (mirroring train_iso_model_free.py)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/iso/monitor", exist_ok=True)
    os.makedirs(f"{args.log_dir}/iso/tensorboard", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.plot_dir}/iso", exist_ok=True)
    os.makedirs("eval_plotsQIwithS_copy/iso", exist_ok=True)
    os.makedirs("tempQIwithS", exist_ok=True)

    # Environment configuration
    env_config = {
        'cost_type': args.cost_type,
        'pricing_policy': args.pricing_policy,
        'demand_pattern': args.demand_pattern,
        'use_dispatch': args.use_dispatch,
        'demand_data': args.demand_data,
        'timesteps': args.timesteps,
        'log_dir': args.log_dir,
        'model_dir': args.model_dir,
        'plot_dir': args.plot_dir,
    }

    print(f"Starting ISO training with {args.iterations} iterations using {args.algo}")
    print(f"Environment config: {env_config}")

    # Create environment
    if args.algo.lower() == "mbpo":
        from energy_net.alternating_wrappers_model_based import make_iso_env_single
        env = make_iso_env_single(
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data
        )
    elif args.algo.lower() == "pets":
        from energy_net.alternating_wrappers_model_based import make_iso_env_single
        env = make_iso_env_single(
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data
        )
    else:
        env = make_env(
            args.pcs_action_sequence,
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data
        )

    print(">>> ISO action space:", env.action_space)
    
    # Save environment normalization for evaluation
    try:
        env.save(f"{args.model_dir}/iso_vecnormalize.pkl")
        print(f"Saved VecNormalize stats to {args.model_dir}/iso_vecnormalize.pkl")
    except Exception:
        pass

    # Initialize or load model
    if args.initial_iso_model and os.path.exists(args.initial_iso_model):
        print(f"\nLoading initial ISO model from {args.initial_iso_model}")
        try:
            # For model-based agents, we need to load the trainer/agent
            # This is a simplified approach - you may need to adjust based on your specific model format
            if args.algo == "mbpo":
                import mbrl.algorithms.mbpo as mbpo
                trainer = mbpo.MBPOTrainer.load(args.initial_iso_model, env=env)
            elif args.algo == "pets":
                from energy_net.pets_adapter import create_pets_trainer
                trainer = create_pets_trainer(env, args)
                if os.path.exists(args.initial_iso_model):
                    trainer.agent.load(args.initial_iso_model)
            else:
                raise ValueError(f"Unknown algorithm for loading: {args.algo}")
            print(f"Successfully loaded {args.algo} model from {args.initial_iso_model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing new model instead...")
            trainer = get_trainer(args.algo, env, args)
    else:
        print(f"\nInitializing new {args.algo} model...")
        trainer = get_trainer(args.algo, env, args)
        
        # Short training step to initialize the model (if supported)
        try:
            if hasattr(trainer, "train"):
                trainer.train(num_epochs=1)
            print("Model initialized successfully")
        except Exception as e:
            print(f"Warning: Could not perform initialization training: {e}")
        
        # Save initial model
        try:
            initial_model_path = f"{args.model_dir}/{args.algo}_iso_init.pt"
            if hasattr(trainer, "save"):
                trainer.save(initial_model_path)
            elif hasattr(trainer, "agent") and hasattr(trainer.agent, "save"):
                trainer.agent.save(initial_model_path)
            print(f"Initial model saved to {initial_model_path}")
        except Exception as e:
            print(f"Warning: Could not save initial model: {e}")

    # Save normalization stats
    try:
        env.save(f"{args.log_dir}/iso/vec_normalize.pkl")
        print(f"Saved VecNormalize stats to {args.log_dir}/iso/vec_normalize.pkl")
    except Exception:
        pass

    # Training loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*20} Iteration {iteration}/{args.iterations} {'='*20}")
        
        # Train the agent
        print(f"\nTraining ISO agent for iteration {iteration}...")
        if hasattr(trainer, "train"):
            results = trainer.train(num_epochs=args.num_epochs)
        else:
            raise RuntimeError("Trainer does not have a train method.")
            
        # Check monitoring after training iteration
        if hasattr(trainer, "monitor") and trainer.monitor is not None:
            trainer.monitor.after_training_iteration()

        # Save the trained model
        model_path = f"{args.model_dir}/{args.algo}_iso_{iteration}.pt"
        if hasattr(trainer, "save"):
            trainer.save(model_path)
        elif hasattr(trainer, "agent") and hasattr(trainer.agent, "save"):
            trainer.agent.save(model_path)
        else:
            print("Warning: No save method found for the trainer or agent.")
        print(f"Model saved to {model_path}")

        # Evaluate current model using a separate evaluation environment
        print("\nEvaluating current ISO model...")
        try:
            pcs_action_sequence = np.load(args.pcs_action_sequence)
        except Exception as e:
            print(f"Warning: Could not load PCS action sequence from {args.pcs_action_sequence}: {e}")
            pcs_action_sequence = None
                # Create a new evaluation environment for each evaluation
        if args.algo.lower() == "mbpo":
            from energy_net.alternating_wrappers_model_based import make_iso_env
            
            print(f"[VERIFICATION] Creating evaluation environment using make_iso_env (CORRECT)")
            print(f"[VERIFICATION] use_dispatch_action={args.use_dispatch}")
            
            eval_env = make_iso_env(
                pcs_action_sequence=pcs_action_sequence,
                steps_per_iteration=args.timesteps,
                cost_type=args.cost_type,
                pricing_policy=args.pricing_policy,
                demand_pattern=args.demand_pattern,
                seed=args.seed + 1000 + iteration,
                log_dir=args.log_dir,
                model_dir=args.model_dir,
                plot_dir=args.plot_dir,
                use_dispatch_action=args.use_dispatch,
                demand_data_path=args.demand_data,
                eval_mode=True,
                norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"  # Use same VecNormalize as training
            )
        elif args.algo.lower() == "pets":
            from energy_net.alternating_wrappers_model_based import make_iso_env
            
            print(f"[VERIFICATION] Creating PETS evaluation environment using make_iso_env")
            print(f"[VERIFICATION] use_dispatch_action={args.use_dispatch}")
            
            eval_env = make_iso_env(
                pcs_action_sequence=pcs_action_sequence,
                steps_per_iteration=args.timesteps,
                cost_type=args.cost_type,
                pricing_policy=args.pricing_policy,
                demand_pattern=args.demand_pattern,
                seed=args.seed + 1000 + iteration,
                log_dir=args.log_dir,
                model_dir=args.model_dir,
                plot_dir=args.plot_dir,
                use_dispatch_action=args.use_dispatch,
                demand_data_path=args.demand_data,
                eval_mode=True,
                norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"  # Use same VecNormalize as training
            )
        else:
            eval_env = make_env(
                args.pcs_action_sequence,
                steps_per_iteration=args.timesteps,
                cost_type=args.cost_type,
                pricing_policy=args.pricing_policy,
                demand_pattern=args.demand_pattern,
                seed=args.seed + 1000 + iteration,  # Different seed for evaluation
                log_dir=args.log_dir,
                model_dir=args.model_dir,
                plot_dir=args.plot_dir,
                use_dispatch_action=args.use_dispatch,
                demand_data_path=args.demand_data,
                eval_mode=True,
                norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"  # Use latest normalization
            )
        eval_results = evaluate_iso(
            agent=trainer.agent if hasattr(trainer, "agent") else trainer,
            env_config=env_config,
            override_env=eval_env,
            pcs_action_sequence=pcs_action_sequence,
            num_episodes=args.eval_episodes,
            seed=args.seed + 1000 + iteration,
            plot_dir=os.path.join(args.plot_dir, f"eval_iter_{iteration}"),
            monitor=trainer.monitor if hasattr(trainer, "monitor") else None
        )
        print(f"Iteration {iteration} evaluation complete!")
        print(f"Average ISO reward: {eval_results['avg_iso_reward']:.2f}")

        # Final evaluation
    print(f"\n{'='*20} Final Evaluation {'='*20}")
    # Create a new evaluation environment for final evaluation
    if args.algo.lower() == "mbpo":
        from energy_net.alternating_wrappers_model_based import make_iso_env
        
        print(f"[VERIFICATION] Creating FINAL evaluation environment using make_iso_env (CORRECT)")
        print(f"[VERIFICATION] use_dispatch_action={args.use_dispatch}")
        
        final_eval_env = make_iso_env(
            pcs_action_sequence=pcs_action_sequence,
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed + 2000,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data,
            eval_mode=True,
            norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"  # Use same VecNormalize as training
        )
    elif args.algo.lower() == "pets":
        from energy_net.alternating_wrappers_model_based import make_iso_env
        
        print(f"[VERIFICATION] Creating FINAL PETS evaluation environment using make_iso_env")
        print(f"[VERIFICATION] use_dispatch_action={args.use_dispatch}")
        
        final_eval_env = make_iso_env(
            pcs_action_sequence=pcs_action_sequence,
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed + 2000,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data,
            eval_mode=True,
            norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"  # Use same VecNormalize as training
        )
    else:
        final_eval_env = make_env(
            args.pcs_action_sequence,
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed + 2000,  # Different seed for final evaluation
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            use_dispatch_action=args.use_dispatch,
            demand_data_path=args.demand_data,
            eval_mode=True,
            norm_path=f"{args.log_dir}/iso/vec_normalize.pkl"
        )
    final_eval_results = evaluate_iso(
        agent=trainer.agent if hasattr(trainer, "agent") else trainer,
        env_config=env_config,
        override_env=final_eval_env,
        pcs_action_sequence=pcs_action_sequence,
        num_episodes=args.eval_episodes,
        seed=args.seed + 2000,
        plot_dir=os.path.join(args.plot_dir, "final_eval"),
        monitor=trainer.monitor if hasattr(trainer, "monitor") else None
    )

    print("\nTraining complete!")
    print(f"Final average ISO reward: {final_eval_results['avg_iso_reward']:.2f}")
    print(f"Final evaluation plots saved to: {os.path.join(args.plot_dir, 'final_eval')}")

    # Save final model
    final_model_path = f"{args.model_dir}/{args.algo}_iso_final.pt"
    if hasattr(trainer, "save"):
        trainer.save(final_model_path)
    elif hasattr(trainer, "agent") and hasattr(trainer.agent, "save"):
        trainer.agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save training results
    results_path = f"{args.log_dir}/training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "final_avg_iso_reward": final_eval_results['avg_iso_reward'],
            "algorithm": args.algo,
            "environment_config": env_config,
            "training_parameters": vars(args)
        }, f, indent=2)
    print(f"Training results saved to {results_path}")


if __name__ == "__main__":
    main() 