# MPC Planning Implementation Documentation

This document provides a comprehensive guide to the Model Predictive Control (MPC) planning system used for goal-conditioned robot control with learned world models.

---

## Table of Contents

1. [Overview](#overview)
2. [Entry Point: run.py](#entry-point-runpy)
3. [Configuration System](#configuration-system)
4. [World Model Architecture (DINOWM)](#world-model-architecture-dinowm)
5. [Cost Model (AutoCostModel)](#cost-model-autocostmodel)
6. [Solver: Cross Entropy Method (CEM)](#solver-cross-entropy-method-cem)
7. [Policy: WorldModelPolicy](#policy-worldmodelpolicy)
8. [Evaluation Pipeline: World.evaluate_from_dataset](#evaluation-pipeline-worldevaluate_from_dataset)
9. [Data Preprocessing](#data-preprocessing)
10. [Execution Flow Summary](#execution-flow-summary)
11. [Shell Script Parameters](#shell-script-parameters)

---

## Overview

The MPC planning system uses a **learned world model** to predict future states given action sequences, then optimizes actions using the **Cross Entropy Method (CEM)** to reach a goal state. The pipeline consists of:

```
run.py → AutoCostModel → WorldModelPolicy → CEMSolver → DINOWM.rollout/get_cost
```

### Key Components

| Component | Description |
|-----------|-------------|
| `DINOWM` | World model that encodes observations and predicts future latent states |
| `CEMSolver` | Optimization algorithm that samples action sequences and selects the best |
| `WorldModelPolicy` | High-level policy wrapper managing action buffers and replanning |
| `World` | Environment wrapper for evaluation |
| `AutoCostModel` | Loads world model checkpoint and provides cost computation |

---

## Entry Point: run.py

**File**: `/home/hnam16/codes/cjepa/plan/run.py`

### Main Function

```python
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def run(cfg):
    # Load world model checkpoint
    path = Path(cfg.wm.path) / f"{cfg.wm.name}_object.ckpt"
    
    # Initialize solver (CEM)
    solver = hydra.utils.instantiate(cfg.solver, action_size=action_size)
    
    # Create policy with planning configuration
    policy = WorldModelPolicy(
        device=device,
        cfg=plan_cfg,
        model=model,      # AutoCostModel
        solver=solver,    # CEMSolver
        img_transform=img_transform,
        action_scaler=action_scaler,
        proprio_scaler=proprio_scaler,
    )
    
    # Run evaluation
    metrics = world.evaluate_from_dataset(
        policy=policy,
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        verbose=True,
        debug=cfg.debug,
    )
```

### Image Transform Function

```python
def img_transform(pixel):
    """Transform raw pixel observations for world model input.
    
    Input: pixel tensor of shape (B, H, W, C) with values in [0, 255]
    Output: normalized tensor of shape (B, C, H, W) with ImageNet normalization
    """
    pixel = pixel.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
    pixel = pixel.float() / 255.0      # [0, 255] → [0, 1]
    
    # Resize and center crop
    pixel = transforms.Resize(196)(pixel)
    pixel = transforms.CenterCrop(196)(pixel)
    
    # ImageNet normalization
    pixel = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )(pixel)
    
    return pixel
```

### Image Transform Details

| Step | Operation | Values |
|------|-----------|--------|
| 1 | Permute | (B, H, W, C) → (B, C, H, W) |
| 2 | Normalize to [0,1] | `/ 255.0` |
| 3 | Resize | Target size: **196** |
| 4 | CenterCrop | Crop size: **196** |
| 5 | ImageNet Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

---

## Configuration System

The configuration is managed by Hydra and consists of multiple YAML files.

### Base Config: config.yaml

**File**: `/home/hnam16/codes/cjepa/plan/config.yaml`

```yaml
defaults:
  - solver: cem          # Uses solver/cem.yaml

horizon: 9               # Planning horizon in steps
receding_horizon: 3      # Steps before replanning
action_block: 3          # Actions per planning step (action chunking)

# Evaluation settings
eval:
  num_eval: 50           # Number of evaluation episodes
  goal_offset_steps: 25  # Steps between initial and goal frames
  eval_budget: 50        # Maximum steps per episode

# World model settings
wm:
  path: null             # Path to checkpoint directory
  name: null             # Model name prefix

debug: false             # Enable debug mode for visualization
```

### Planning Configuration Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 9 | Number of prediction steps in the planning horizon |
| `receding_horizon` | 3 | Execute this many steps before replanning |
| `action_block` | 3 | Number of actions grouped together (action chunking) |
| `plan_len` | `horizon * action_block` = 27 | Total action sequence length |

### CEM Solver Config: solver/cem.yaml

**File**: `/home/hnam16/codes/cjepa/plan/solver/cem.yaml`

```yaml
_target_: stable_worldmodel.solver.CEMSolver

num_samples: 300    # Number of action sequences sampled per iteration
var_scale: 1.0      # Initial variance scaling factor
n_steps: 30         # Number of CEM optimization iterations
topk: 30            # Number of top samples for distribution update
```

### CEM Solver Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_samples` | **300** | Candidate action sequences sampled each iteration |
| `var_scale` | **1.0** | Initial variance multiplier |
| `n_steps` | **30** | Optimization iterations |
| `topk` | **30** | Elite samples for mean/variance update |
| Selection ratio | 30/300 = **10%** | Top 10% of samples used for updates |

---

## World Model Architecture (DINOWM)

**File**: `/home/hnam16/codes/cjepa/stable-worldmodel/stable_worldmodel/wm/dinowm.py`

DINOWM is a latent dynamics model that operates on visual and proprioceptive observations.

### Class Structure

```python
class DINOWM(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,           # Visual encoder (slot-based)
        predictor: nn.Module,         # Temporal prediction model
        action_encoder: nn.Module,    # Action embedding network
        proprio_encoder: nn.Module,   # Proprioception embedding network
        proprio_loss_type: str = "mse",
        proprio_loss_scale: float = 1.0,
        lr: float = 1e-4,
        slot_agg_type: str = "concat",  # How to aggregate slot features
    ):
```

### Key Methods

#### 1. `encode(pixel, proprio)`

Encodes visual observations and proprioception into latent representations.

```python
def encode(self, pixel, proprio):
    """Encode observations into latent space.
    
    Args:
        pixel: (B, T, C, H, W) or (B, C, H, W) visual observations
        proprio: (B, T, D) or (B, D) proprioceptive state
    
    Returns:
        embed: (B, T, num_slots, slot_dim) slot-based visual embeddings
        proprio_embed: (B, T, proprio_dim) proprioceptive embeddings
    """
    # Visual encoding through slot attention
    slots = self.encoder(pixel)  # (B*T, num_slots, slot_dim)
    
    # Proprioception encoding
    proprio_embed = self.proprio_encoder(proprio)
    
    return slots, proprio_embed
```

#### 2. `predict(embed, proprio_embed, action)`

Predicts next state given current state and action.

```python
def predict(self, embed, proprio_embed, action):
    """Predict next latent state.
    
    Args:
        embed: (B, T, num_slots, slot_dim) current visual embedding
        proprio_embed: (B, T, proprio_dim) current proprio embedding
        action: (B, T, action_dim) action sequence
    
    Returns:
        next_embed: (B, T, num_slots, slot_dim) predicted visual embedding
        next_proprio_embed: (B, T, proprio_dim) predicted proprio embedding
    """
    # Encode action
    action_embed = self.action_encoder(action)
    
    # Predict through transformer predictor
    next_embed, next_proprio_embed = self.predictor(
        embed, proprio_embed, action_embed
    )
    
    return next_embed, next_proprio_embed
```

#### 3. `rollout(embed, proprio_embed, action)`

**Critical for MPC**: Iteratively predicts future states over action sequence.

```python
def rollout(self, embed, proprio_embed, action):
    """Rollout predictions over action sequence.
    
    Args:
        embed: (B, num_slots, slot_dim) initial visual embedding
        proprio_embed: (B, proprio_dim) initial proprio embedding  
        action: (B, horizon, action_dim) action sequence
    
    Returns:
        all_embeds: (B, horizon+1, num_slots, slot_dim) predicted trajectory
        all_proprio: (B, horizon+1, proprio_dim) predicted proprio trajectory
    """
    B, horizon, _ = action.shape
    
    # Initialize trajectory storage
    all_embeds = [embed]
    all_proprio = [proprio_embed]
    
    current_embed = embed
    current_proprio = proprio_embed
    
    # Iterative prediction
    for t in range(horizon):
        # Get action for this timestep
        action_t = action[:, t:t+1, :]  # (B, 1, action_dim)
        
        # Predict next state
        next_embed, next_proprio = self.predict(
            current_embed.unsqueeze(1),
            current_proprio.unsqueeze(1),
            action_t
        )
        
        # Update current state
        current_embed = next_embed.squeeze(1)
        current_proprio = next_proprio.squeeze(1)
        
        # Store predictions
        all_embeds.append(current_embed)
        all_proprio.append(current_proprio)
    
    # Stack all predictions
    all_embeds = torch.stack(all_embeds, dim=1)    # (B, horizon+1, ...)
    all_proprio = torch.stack(all_proprio, dim=1)  # (B, horizon+1, ...)
    
    return all_embeds, all_proprio
```

#### 4. `get_cost(pred_embed, pred_proprio, goal_embed, goal_proprio)`

**Cost function for MPC optimization**.

```python
def get_cost(self, pred_embed, pred_proprio, goal_embed, goal_proprio):
    """Compute cost between predicted and goal states.
    
    Args:
        pred_embed: (B, horizon, num_slots, slot_dim) predicted visual embeddings
        pred_proprio: (B, horizon, proprio_dim) predicted proprio embeddings
        goal_embed: (B, num_slots, slot_dim) goal visual embedding
        goal_proprio: (B, proprio_dim) goal proprio embedding
    
    Returns:
        cost: (B,) scalar cost for each sample
    """
    B, horizon, num_slots, slot_dim = pred_embed.shape
    
    # Expand goal to match prediction shape
    goal_embed_exp = goal_embed.unsqueeze(1).expand(-1, horizon, -1, -1)
    goal_proprio_exp = goal_proprio.unsqueeze(1).expand(-1, horizon, -1)
    
    # Aggregate slots: concatenate all slots into single vector
    if self.slot_agg_type == "concat":
        pred_flat = pred_embed.reshape(B, horizon, -1)      # (B, H, num_slots*slot_dim)
        goal_flat = goal_embed_exp.reshape(B, horizon, -1)
    
    # MSE loss for visual embedding
    pixel_cost = F.mse_loss(pred_flat, goal_flat, reduction='none')
    pixel_cost = pixel_cost.mean(dim=-1)  # (B, horizon)
    
    # MSE loss for proprioception
    proprio_cost = F.mse_loss(pred_proprio, goal_proprio_exp, reduction='none')
    proprio_cost = proprio_cost.mean(dim=-1)  # (B, horizon)
    
    # Total cost: sum over horizon, take mean
    total_cost = (pixel_cost + proprio_cost).mean(dim=1)  # (B,)
    
    return total_cost
```

### Cost Function Details

| Component | Computation | Reduction |
|-----------|-------------|-----------|
| Visual cost | MSE(pred_embed, goal_embed) | mean over slot_dim → mean over horizon |
| Proprio cost | MSE(pred_proprio, goal_proprio) | mean over proprio_dim → mean over horizon |
| Total cost | visual + proprio | final scalar per sample |

---

## Cost Model (AutoCostModel)

**File**: `/home/hnam16/codes/cjepa/stable-worldmodel/stable_worldmodel/policy.py`

Wrapper that loads a world model checkpoint and extracts the cost-computation module.

```python
class AutoCostModel:
    """Automatically load cost model from checkpoint."""
    
    def __init__(self, path: str):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file (e.g., "model_name_object.ckpt")
        
        The checkpoint file should end with "_object.ckpt" and contains
        the DINOWM module with rollout and get_cost methods.
        """
        self.model = self._load_model(path)
    
    def _load_model(self, path):
        # Load checkpoint
        ckpt = torch.load(path, map_location='cpu')
        
        # Extract model state
        model = ckpt['model']  # or reconstruct from state_dict
        
        return model
    
    def rollout(self, *args, **kwargs):
        return self.model.rollout(*args, **kwargs)
    
    def get_cost(self, *args, **kwargs):
        return self.model.get_cost(*args, **kwargs)
```

### Checkpoint Naming Convention

```
{model_name}_object.ckpt
```

Example: `pusht_causal_wm_52_epoch_10_object.ckpt`

---

## Solver: Cross Entropy Method (CEM)

**File**: `/home/hnam16/codes/cjepa/stable-worldmodel/stable_worldmodel/solver/cem.py`

CEM is a gradient-free optimization algorithm that iteratively refines action distributions.

### Class Definition

```python
class CEMSolver:
    """Cross Entropy Method solver for action optimization."""
    
    def __init__(
        self,
        action_size: int,         # Dimension of action space
        num_samples: int = 300,   # Number of samples per iteration
        n_steps: int = 30,        # Number of optimization steps
        topk: int = 30,           # Number of elite samples
        var_scale: float = 1.0,   # Initial variance scale
    ):
        self.action_size = action_size
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.var_scale = var_scale
```

### Key Methods

#### 1. `init_action_distrib(plan_len)`

Initialize the action distribution.

```python
def init_action_distrib(self, plan_len: int):
    """Initialize Gaussian action distribution.
    
    Args:
        plan_len: Total length of action sequence (horizon * action_block)
    
    Returns:
        mean: (plan_len, action_size) initial mean (zeros)
        var: (plan_len, action_size) initial variance (ones * var_scale)
    """
    mean = torch.zeros(plan_len, self.action_size)
    var = torch.ones(plan_len, self.action_size) * self.var_scale
    return mean, var
```

#### 2. `solve(cost_fn, mean, var)`

Main optimization loop.

```python
def solve(self, cost_fn, mean, var):
    """Optimize action sequence using CEM.
    
    Args:
        cost_fn: Function (actions) → costs, where
                 actions: (num_samples, plan_len, action_size)
                 costs: (num_samples,)
        mean: (plan_len, action_size) current mean
        var: (plan_len, action_size) current variance
    
    Returns:
        best_action: (plan_len, action_size) optimized action sequence
        final_mean: (plan_len, action_size) final distribution mean
        final_var: (plan_len, action_size) final distribution variance
    """
    for step in range(self.n_steps):
        # Step 1: Sample action sequences from current distribution
        # Sample from N(mean, var) for each timestep
        noise = torch.randn(self.num_samples, *mean.shape)  # (300, plan_len, action_size)
        actions = mean + noise * torch.sqrt(var)            # (300, plan_len, action_size)
        
        # Step 2: Evaluate costs for all samples
        costs = cost_fn(actions)  # (300,)
        
        # Step 3: Select top-k samples (elite set)
        _, topk_indices = torch.topk(costs, self.topk, largest=False)  # smallest costs
        elite_actions = actions[topk_indices]  # (30, plan_len, action_size)
        
        # Step 4: Update distribution from elite samples
        mean = elite_actions.mean(dim=0)  # (plan_len, action_size)
        var = elite_actions.var(dim=0)    # (plan_len, action_size)
    
    # Return best action (mean of final distribution)
    best_action = mean
    
    return best_action, mean, var
```

### CEM Algorithm Visualization

```
Iteration 1:
  Sample 300 actions from N(0, 1)
  Evaluate costs via world model rollout
  Select top 30 (lowest cost)
  Update mean/var from elite samples

Iteration 2:
  Sample 300 actions from N(mean₁, var₁)
  Evaluate costs
  Select top 30
  Update mean/var

... repeat for 30 iterations ...

Final: Return mean as optimal action sequence
```

### CEM Numerical Summary

| Step | Operation | Shape |
|------|-----------|-------|
| Sample | `N(mean, var)` | (300, plan_len, action_size) |
| Evaluate | `cost_fn(actions)` | (300,) |
| Select | `topk(costs, k=30)` | (30, plan_len, action_size) |
| Update mean | `elite.mean(dim=0)` | (plan_len, action_size) |
| Update var | `elite.var(dim=0)` | (plan_len, action_size) |

---

## Policy: WorldModelPolicy

**File**: `/home/hnam16/codes/cjepa/stable-worldmodel/stable_worldmodel/policy.py`

High-level policy that manages action buffers and replanning logic.

### PlanConfig Dataclass

```python
@dataclass
class PlanConfig:
    """Configuration for planning."""
    
    horizon: int = 9          # Planning horizon in prediction steps
    receding_horizon: int = 3 # Steps to execute before replanning
    history_len: int = 1      # Number of past observations to use
    action_block: int = 3     # Actions per prediction step (chunking)
    warm_start: bool = True   # Use previous solution to initialize next plan
    
    @property
    def plan_len(self) -> int:
        """Total action sequence length."""
        return self.horizon * self.action_block
```

### Planning Length Calculation

```
plan_len = horizon × action_block

Example:
  horizon = 9
  action_block = 3
  plan_len = 9 × 3 = 27 actions
```

### WorldModelPolicy Class

```python
class WorldModelPolicy:
    """Policy using world model for planning."""
    
    def __init__(
        self,
        device: str,
        cfg: PlanConfig,
        model: AutoCostModel,        # World model with rollout/get_cost
        solver: CEMSolver,           # CEM optimizer
        img_transform: Callable,     # Image preprocessing function
        action_scaler: StandardScaler,   # Action normalization
        proprio_scaler: StandardScaler,  # Proprioception normalization
    ):
        self.device = device
        self.cfg = cfg
        self.model = model
        self.solver = solver
        self.img_transform = img_transform
        self.action_scaler = action_scaler
        self.proprio_scaler = proprio_scaler
        
        # Action buffer for receding horizon
        self.action_buffer = None
        self.buffer_idx = 0
        
        # Distribution state for warm start
        self.mean = None
        self.var = None
```

### Key Methods

#### 1. `reset()`

Reset policy state for new episode.

```python
def reset(self):
    """Reset action buffer and distribution state."""
    self.action_buffer = None
    self.buffer_idx = 0
    self.mean = None
    self.var = None
```

#### 2. `set_goal(goal_pixel, goal_proprio)`

Set target goal state.

```python
def set_goal(self, goal_pixel, goal_proprio):
    """Set goal state for planning.
    
    Args:
        goal_pixel: (H, W, C) goal image observation
        goal_proprio: (D,) goal proprioceptive state
    """
    # Preprocess goal observation
    goal_pixel = self.img_transform(goal_pixel.unsqueeze(0))  # (1, C, H, W)
    goal_proprio = self.proprio_scaler.transform(goal_proprio)
    
    # Encode goal into latent space
    with torch.no_grad():
        self.goal_embed, self.goal_proprio_embed = self.model.encode(
            goal_pixel.to(self.device),
            goal_proprio.to(self.device)
        )
```

#### 3. `get_action(obs)`

Main action selection method.

```python
def get_action(self, obs):
    """Get action for current observation.
    
    Args:
        obs: dict with 'pixels' and 'agent_pos' keys
    
    Returns:
        action: (action_dim,) action to execute
    """
    # Check if we need to replan
    if self._should_replan():
        self._plan(obs)
    
    # Get action from buffer
    action = self.action_buffer[self.buffer_idx]
    self.buffer_idx += 1
    
    # Denormalize action for environment
    action = self.action_scaler.inverse_transform(action)
    
    return action

def _should_replan(self):
    """Check if replanning is needed."""
    if self.action_buffer is None:
        return True
    if self.buffer_idx >= self.cfg.receding_horizon * self.cfg.action_block:
        return True
    return False
```

#### 4. `_plan(obs)`

Execute CEM optimization to get action plan.

```python
def _plan(self, obs):
    """Plan action sequence using CEM.
    
    Args:
        obs: Current observation dict
    """
    # Preprocess observation
    pixel = self.img_transform(obs['pixels'].unsqueeze(0))
    proprio = self.proprio_scaler.transform(obs['agent_pos'])
    
    # Encode current state
    with torch.no_grad():
        current_embed, current_proprio_embed = self.model.encode(
            pixel.to(self.device),
            proprio.to(self.device)
        )
    
    # Initialize or warm start distribution
    if self.mean is None or not self.cfg.warm_start:
        self.mean, self.var = self.solver.init_action_distrib(self.cfg.plan_len)
    else:
        # Shift distribution for warm start (remove executed actions)
        shift = self.cfg.receding_horizon * self.cfg.action_block
        self.mean = torch.cat([self.mean[shift:], torch.zeros(shift, self.mean.shape[1])], dim=0)
        self.var = torch.cat([self.var[shift:], torch.ones(shift, self.var.shape[1])], dim=0)
    
    # Define cost function for CEM
    def cost_fn(actions):
        """Evaluate cost of action sequences.
        
        Args:
            actions: (num_samples, plan_len, action_dim) sampled actions
        
        Returns:
            costs: (num_samples,) cost for each sequence
        """
        B = actions.shape[0]
        
        # Expand current state for all samples
        embed = current_embed.expand(B, -1, -1)
        proprio_embed = current_proprio_embed.expand(B, -1)
        
        # Reshape actions for horizon steps
        # (B, plan_len, action_dim) → (B, horizon, action_block, action_dim)
        actions = actions.view(B, self.cfg.horizon, self.cfg.action_block, -1)
        
        # For prediction, we typically use the last action in each block
        # or aggregate them somehow
        actions = actions[:, :, -1, :]  # (B, horizon, action_dim)
        
        # Rollout world model
        with torch.no_grad():
            pred_embed, pred_proprio = self.model.rollout(
                embed, proprio_embed, actions
            )
        
        # Compute cost to goal
        costs = self.model.get_cost(
            pred_embed[:, 1:],  # Skip initial state
            pred_proprio[:, 1:],
            self.goal_embed.expand(B, -1, -1),
            self.goal_proprio_embed.expand(B, -1)
        )
        
        return costs
    
    # Run CEM optimization
    best_action, self.mean, self.var = self.solver.solve(
        cost_fn, self.mean.to(self.device), self.var.to(self.device)
    )
    
    # Store action buffer
    self.action_buffer = best_action
    self.buffer_idx = 0
```

### Receding Horizon Diagram

```
Plan (27 actions total, action_block=3, horizon=9):
[a₁][a₂][a₃] | [a₄][a₅][a₆] | [a₇][a₈][a₉] | ... | [a₂₅][a₂₆][a₂₇]
 ↑──────────↑
 Execute 9 actions (receding_horizon=3 × action_block=3)

After executing 9 actions → REPLAN with new observation
```

---

## Evaluation Pipeline: World.evaluate_from_dataset

**File**: `/home/hnam16/codes/cjepa/stable-worldmodel/stable_worldmodel/world.py`

### World Class Overview

```python
class World:
    """High-level environment manager for evaluation."""
    
    def __init__(
        self,
        env_id: str = "swm/PushT-v1",
        num_env: int = 1,
        device: str = "cuda",
    ):
        self.env = make_envs(env_id, num_env)
        self.device = device
```

### evaluate_from_dataset Method

```python
def evaluate_from_dataset(
    self,
    policy: WorldModelPolicy,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    verbose: bool = True,
    debug: bool = False,
):
    """Evaluate policy on episodes from dataset.
    
    Args:
        policy: WorldModelPolicy instance
        goal_offset_steps: Number of steps between initial and goal frame
        eval_budget: Maximum steps allowed per episode
        verbose: Print progress
        debug: Enable visualization
    
    Returns:
        metrics: dict with evaluation metrics
    
    Pipeline:
        1. Load episode data from dataset
        2. Set initial state from data
        3. Set goal state (goal_offset_steps ahead)
        4. Run policy for eval_budget steps
        5. Compute success metrics
    """
    metrics_list = []
    
    for episode_idx in range(self.num_eval):
        # 1. Load episode data
        episode = self.dataset[episode_idx]
        
        # 2. Set initial environment state
        init_pixel = episode['pixels'][0]           # First frame
        init_proprio = episode['agent_pos'][0]      # First agent position
        self.env.reset()
        self.env.set_state(init_proprio)            # Teleport to initial state
        
        # 3. Set goal state (goal_offset_steps into the future)
        goal_idx = goal_offset_steps
        goal_pixel = episode['pixels'][goal_idx]
        goal_proprio = episode['agent_pos'][goal_idx]
        
        # 4. Initialize policy with goal
        policy.reset()
        policy.set_goal(goal_pixel, goal_proprio)
        
        # 5. Run policy for eval_budget steps
        total_reward = 0
        for step in range(eval_budget):
            # Get current observation
            obs = self.env.get_obs()
            
            # Get action from policy
            action = policy.get_action(obs)
            
            # Execute action in environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # 6. Compute final metrics
        final_distance = compute_distance(
            obs['agent_pos'], goal_proprio
        )
        success = final_distance < self.success_threshold
        
        metrics_list.append({
            'reward': total_reward,
            'success': success,
            'final_distance': final_distance,
            'steps': step + 1,
        })
    
    # Aggregate metrics
    metrics = {
        'mean_reward': np.mean([m['reward'] for m in metrics_list]),
        'success_rate': np.mean([m['success'] for m in metrics_list]),
        'mean_distance': np.mean([m['final_distance'] for m in metrics_list]),
    }
    
    return metrics
```

### Evaluation Flow Diagram

```
Episode n:
┌─────────────────────────────────────────────────────────────────┐
│ Dataset: [frame₀] [frame₁] ... [frame₂₄] [frame₂₅] ... [frame_T]│
│              ↑                              ↑                    │
│           INITIAL                         GOAL                   │
│         (step 0)                    (offset 25)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Environment Execution:                                           │
│  t=0: obs → policy.get_action() → action → env.step()          │
│  t=1: obs → policy.get_action() → action → env.step()          │
│  ...                                                             │
│  t=49: obs → policy.get_action() → action → env.step()         │
│                                                                  │
│  (Every receding_horizon×action_block steps: REPLAN)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Metrics: distance to goal, success rate, total reward           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

### Action and Proprioception Scaling

Both actions and proprioception are normalized using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

# Fit scalers on training data
action_scaler = StandardScaler()
action_scaler.fit(training_actions)  # Computes mean and std

proprio_scaler = StandardScaler()
proprio_scaler.fit(training_proprio)

# Transform: (x - mean) / std
normalized_action = action_scaler.transform(raw_action)
normalized_proprio = proprio_scaler.transform(raw_proprio)

# Inverse transform: x * std + mean
raw_action = action_scaler.inverse_transform(normalized_action)
```

### StandardScaler Statistics (Loaded from Training)

The scalers are fitted on training data and saved with the model checkpoint. For PushT:

| Variable | Dimensions | Example Mean | Example Std |
|----------|------------|--------------|-------------|
| Action | 2 (dx, dy) | ~0.0 | ~0.1-0.5 |
| Proprio | 2 (x, y) | ~256 | ~100 |

### Image Normalization

**ImageNet Statistics** (from `dataset_stats.py`):

```python
ImageNet = dict(
    mean=[0.485, 0.456, 0.406],  # RGB channel means
    std=[0.229, 0.224, 0.225]    # RGB channel stds
)
```

### Complete Preprocessing Pipeline

```
Raw Observation:
├── pixel: (H, W, C) uint8 [0, 255]
│   ├── permute → (C, H, W)
│   ├── / 255.0 → [0, 1]
│   ├── Resize(196)
│   ├── CenterCrop(196)
│   └── Normalize(ImageNet) → (C, 196, 196) float32
│
└── proprio: (D,) float32
    └── StandardScaler → normalized float32
```

---

## Execution Flow Summary

### Complete MPC Loop

```
1. INITIALIZATION
   ├── Load world model from checkpoint: {name}_object.ckpt
   ├── Initialize CEM solver (300 samples, 30 steps, top 30)
   ├── Create WorldModelPolicy with PlanConfig
   └── Load StandardScalers for action/proprio

2. FOR EACH EPISODE:
   ├── Load initial/goal frames from dataset
   ├── Reset environment to initial state
   ├── Encode goal observation → goal_embed
   └── policy.reset()

3. FOR EACH STEP (up to eval_budget=50):
   │
   ├── IF should_replan (every receding_horizon×action_block steps):
   │   │
   │   ├── Encode current observation → current_embed
   │   │
   │   ├── Initialize/warm-start action distribution
   │   │
   │   └── CEM OPTIMIZATION (30 iterations):
   │       │
   │       ├── Sample 300 action sequences from N(mean, var)
   │       │
   │       ├── FOR EACH SAMPLE:
   │       │   ├── world_model.rollout(current_embed, actions)
   │       │   └── cost = MSE(predicted_embed, goal_embed)
   │       │
   │       ├── Select top 30 lowest-cost samples
   │       │
   │       └── Update mean, var from elite samples
   │   
   ├── Get action from buffer
   ├── Denormalize action
   ├── Execute action in environment
   └── Record reward

4. COMPUTE METRICS
   ├── Final distance to goal
   ├── Success rate
   └── Mean reward
```

### Timing Breakdown (Approximate)

| Component | Operations per Episode | Per-Step Time |
|-----------|----------------------|---------------|
| CEM Optimization | ~eval_budget/receding_horizon ≈ 17 | ~1-2s |
| Each CEM iteration | 300 forward passes × 30 steps | ~30ms |
| World model rollout | horizon × predict calls | ~5ms |
| Environment step | 1 | <1ms |

---

## Shell Script Parameters

**File**: `/home/hnam16/codes/cjepa/scripts/pusht/run_planning.sh`

### Example Configuration

```bash
EXPNUM="263p"
HORIZON=5
RECEDING_HORIZON=5
ACTION_BLOCK=5

# Evaluation settings
EVAL_BUDGET=50
GOAL_OFFSET_STEPS=25
NUM_EVAL=50

# World model path
WM_PATH="/path/to/checkpoint"
WM_NAME="pusht_causal_wm_52_epoch_10"
```

### Parameter Relationships

| Shell Variable | Config Key | Value | Computed |
|----------------|------------|-------|----------|
| HORIZON | horizon | 5 | - |
| RECEDING_HORIZON | receding_horizon | 5 | - |
| ACTION_BLOCK | action_block | 5 | - |
| - | plan_len | - | 5 × 5 = **25** |
| EVAL_BUDGET | eval.eval_budget | 50 | - |
| GOAL_OFFSET_STEPS | eval.goal_offset_steps | 25 | - |
| NUM_EVAL | eval.num_eval | 50 | - |

### Hydra Command

```bash
python plan/run.py \
    horizon=${HORIZON} \
    receding_horizon=${RECEDING_HORIZON} \
    action_block=${ACTION_BLOCK} \
    eval.eval_budget=${EVAL_BUDGET} \
    eval.goal_offset_steps=${GOAL_OFFSET_STEPS} \
    eval.num_eval=${NUM_EVAL} \
    wm.path=${WM_PATH} \
    wm.name=${WM_NAME}
```

---

## Quick Reference: All Hyperparameters

### Planning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 9 (or 5) | Number of prediction steps |
| `receding_horizon` | 3 (or 5) | Steps before replanning |
| `action_block` | 3 (or 5) | Actions per step (chunking) |
| `plan_len` | 27 (or 25) | `horizon × action_block` |
| `history_len` | 1 | Past observations used |
| `warm_start` | True | Reuse previous solution |

### CEM Solver

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_samples` | 300 | Candidates per iteration |
| `n_steps` | 30 | Optimization iterations |
| `topk` | 30 | Elite samples for update |
| `var_scale` | 1.0 | Initial variance |

### Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_eval` | 50 | Episodes to evaluate |
| `eval_budget` | 50 | Max steps per episode |
| `goal_offset_steps` | 25 | Steps to goal frame |

### Image Processing

| Step | Value |
|------|-------|
| Resize | 196 |
| CenterCrop | 196 |
| Mean (RGB) | [0.485, 0.456, 0.406] |
| Std (RGB) | [0.229, 0.224, 0.225] |

---

## Summary

The MPC planning system combines:

1. **DINOWM World Model**: Slot-based visual encoder with temporal prediction
2. **CEM Solver**: 300 samples, 30 iterations, top-30 selection
3. **Receding Horizon Control**: Replan every N steps with warm start
4. **Goal-Conditioned Cost**: MSE between predicted and goal embeddings

This enables the agent to reach goal states specified as images by optimizing action sequences through learned dynamics.
