from types import SimpleNamespace
import numpy as np
import gymnasium as gym
import torch

def make_sac(num_inputs, action_space, args):
    """
    Factory for mbrl.third_party.pytorch_sac_pranz24.SAC that accepts Hydra kwargs
    and calls SAC with positional arguments.
    Expects:
      - num_inputs: int
      - action_space: dict with keys low, high, shape
      - args: dict of hyperparameters; must include hidden_size
    """
    from mbrl.third_party.pytorch_sac_pranz24 import SAC

    # Build a proper Box action space from dict and ensure 1D shape
    raw_shape = action_space.get("shape")
    if raw_shape is None:
        # fallback: infer from lows/highs if present
        if action_space.get("low") is not None:
            raw_shape = np.array(action_space["low"]).shape
        elif action_space.get("high") is not None:
            raw_shape = np.array(action_space["high"]).shape
        else:
            raw_shape = (1,)
    # Normalize to tuple
    if isinstance(raw_shape, int):
        raw_shape = (raw_shape,)
    raw_shape = tuple(raw_shape)
    # Compute flat action dim (handle scalar -> 1)
    action_dim = int(np.prod(raw_shape)) if len(raw_shape) > 0 else 1
    shape = (action_dim,)

    # Low/high defaults and reshape to (action_dim,)
    if action_space.get("low") is not None:
        low = np.array(action_space["low"], dtype=np.float32).reshape(-1)
    else:
        low = -np.ones(action_dim, dtype=np.float32)
    if action_space.get("high") is not None:
        high = np.array(action_space["high"], dtype=np.float32).reshape(-1)
    else:
        high = np.ones(action_dim, dtype=np.float32)
    # In case provided arrays have mismatched length, broadcast/clip
    if low.size != action_dim:
        low = np.full(action_dim, low.min() if low.size > 0 else -1.0, dtype=np.float32)
    if high.size != action_dim:
        high = np.full(action_dim, high.max() if high.size > 0 else 1.0, dtype=np.float32)

    box = gym.spaces.Box(low=low.reshape(shape), high=high.reshape(shape), dtype=np.float32)

    # Convert args dict to a namespace, ensure required fields exist
    args_dict = dict(args or {})
    # Required policy type for SAC implementation
    args_dict.setdefault("policy", "Gaussian")
    # Hidden sizes
    if "hidden_size" not in args_dict:
        hs = args_dict.get("actor_hidden_dim") or args_dict.get("critic_hidden_dim") or 256
        args_dict["hidden_size"] = hs
    # Common hyperparameters defaults
    args_dict.setdefault("lr", 3e-4)
    args_dict.setdefault("tau", 0.005)
    args_dict.setdefault("gamma", 0.99)
    args_dict.setdefault("target_update_interval", 4)
    args_dict.setdefault("automatic_entropy_tuning", True)
    args_dict.setdefault("target_entropy", -0.05)
    # Device
    args_dict.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")

    sac_args = SimpleNamespace(**args_dict)

    return SAC(num_inputs, box, sac_args)
