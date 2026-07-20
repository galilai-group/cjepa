"""C-JEPA model and its object-masked predictor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torch import nn


class NonCausalTransformer(nn.Module):
    """The full-attention transformer used by the original C-JEPA predictor."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            dim, heads, dropout=dropout, batch_first=True
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, dim),
                            nn.Dropout(dropout),
                        ),
                    ]
                )
            )

    def forward(self, x, return_attention=False):
        weights = [] if return_attention else None
        for attention, feed_forward in self.layers:
            if return_attention:
                attention_out, attention_weights = attention(
                    x, x, x, need_weights=True
                )
                weights.append(attention_weights)
            else:
                # Preserve the original checkpoint's execution path.  Its
                # MultiheadAttention call used the default need_weights=True.
                attention_out, _ = attention(x, x, x)
            x = x + attention_out
            x = x + feed_forward(x)
        output = self.norm(x)
        return (output, weights) if return_attention else output


class MaskedSlotPredictor(nn.Module):
    """Predict future slots while reconstructing masked object trajectories."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        history_frames: int = 3,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        seed: int = 42,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        num_unmaskable_slots: int = 0,
    ):
        super().__init__()
        if not 0 <= num_masked_slots <= num_slots - num_unmaskable_slots:
            raise ValueError("num_masked_slots exceeds the object-slot count")

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        self.num_unmaskable_slots = num_unmaskable_slots
        self.seed = seed

        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, slot_dim)
        )
        self.id_projector = nn.Linear(slot_dim, slot_dim)
        self.transformer = NonCausalTransformer(
            dim=slot_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.to_out = nn.Linear(slot_dim, slot_dim)

    def get_mask_indices(self, batch_size, device):
        del batch_size
        rng = np.random.RandomState(self.seed)
        object_slots = self.num_slots - self.num_unmaskable_slots
        masked = rng.choice(object_slots, self.num_masked_slots, replace=False)
        is_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_masked[masked] = True
        return is_masked, torch.from_numpy(masked).to(device)

    def prepare_input(self, x):
        batch_size, history_frames, num_slots, slot_dim = x.shape
        if history_frames > self.history_frames:
            raise ValueError(
                f"Expected at most {self.history_frames} history frames, "
                f"got {history_frames}"
            )

        if self.num_masked_slots:
            is_masked, masked_indices = self.get_mask_indices(batch_size, x.device)
        else:
            is_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=x.device)
            masked_indices = torch.empty(0, dtype=torch.long, device=x.device)

        anchors = x[:, 0]
        anchor_queries = self.id_projector(anchors)
        total_frames = history_frames + self.pred_frames
        time_positions = self.time_pos_embed[:, -total_frames:]
        query = (
            self.mask_token.expand(batch_size, total_frames, num_slots, slot_dim)
            + time_positions.expand(batch_size, total_frames, num_slots, slot_dim)
            + anchor_queries.unsqueeze(1).expand(
                batch_size, total_frames, num_slots, slot_dim
            )
        )

        final_input = query.clone()
        final_input[:, 0] = x[:, 0] + time_positions[:, 0]
        visible_indices = torch.where(~is_masked)[0]
        if len(visible_indices) and history_frames > 1:
            positions = time_positions[:, 1:history_frames].expand(
                batch_size,
                history_frames - 1,
                num_slots,
                slot_dim,
            )
            final_input[:, 1:history_frames, visible_indices] = (
                x[:, 1:, visible_indices] + positions[:, :, visible_indices]
            )
        return final_input, masked_indices

    def forward(self, x):
        model_input, masked_indices = self.prepare_input(x)
        output = self.transformer(rearrange(model_input, "b t s d -> b (t s) d"))
        output = rearrange(
            output,
            "b (t s) d -> b t s d",
            t=model_input.shape[1],
            s=model_input.shape[2],
        )
        return self.to_out(output), masked_indices

    @torch.no_grad()
    def inference(self, x):
        batch_size, history_frames, num_slots, slot_dim = x.shape
        total_frames = history_frames + self.pred_frames
        time_positions = self.time_pos_embed[:, -total_frames:]
        anchors = self.id_projector(x[:, 0])

        history = x + time_positions[:, :history_frames]
        future = (
            self.mask_token.expand(batch_size, self.pred_frames, num_slots, slot_dim)
            + time_positions[:, history_frames:].expand(
                batch_size, self.pred_frames, num_slots, slot_dim
            )
            + anchors.unsqueeze(1).expand(
                batch_size, self.pred_frames, num_slots, slot_dim
            )
        )
        output = self.transformer(
            rearrange(
                torch.cat([history, future], dim=1),
                "b t s d -> b (t s) d",
            )
        )
        output = rearrange(
            output,
            "b (t s) d -> b t s d",
            t=total_frames,
            s=num_slots,
        )
        return self.to_out(output)[:, history_frames:]


class NodeEmbedder(nn.Module):
    """Embed one action or proprioception vector as one slot-sized node."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, values):
        values = values.float().transpose(1, 2)
        return self.projection(values).transpose(1, 2)


class CJEPA(nn.Module):
    """C-JEPA plus the small adapter required by stable-worldmodel.

    The object encoder is optional. If a batch contains a ``slots`` column,
    those pre-extracted representations are used directly.
    """

    def __init__(
        self,
        num_object_slots: int,
        slot_dim: int = 128,
        history_frames: int = 3,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        seed: int = 42,
        action_dim: int = 0,
        action_frameskip: int = 1,
        proprio_dim: int = 0,
        object_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.num_object_slots = num_object_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.action_dim = action_dim
        self.action_frameskip = action_frameskip
        self.proprio_dim = proprio_dim
        self.object_encoder = object_encoder

        self.proprio_encoder = (
            NodeEmbedder(proprio_dim, slot_dim) if proprio_dim else None
        )
        self.action_encoder = (
            NodeEmbedder(action_dim * action_frameskip, slot_dim)
            if action_dim
            else None
        )
        extra_nodes = int(self.proprio_encoder is not None) + int(
            self.action_encoder is not None
        )
        self.predictor = MaskedSlotPredictor(
            num_slots=num_object_slots + extra_nodes,
            slot_dim=slot_dim,
            history_frames=history_frames,
            pred_frames=pred_frames,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_unmaskable_slots=extra_nodes,
        )
        self._planning_cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def action_node(self) -> int | None:
        if self.action_encoder is None:
            return None
        return self.num_object_slots + int(self.proprio_encoder is not None)

    @property
    def proprio_node(self) -> int | None:
        return self.num_object_slots if self.proprio_encoder is not None else None

    def encode_objects(self, batch, initial_slots=None, return_last_slot=False):
        if "slots" in batch:
            slots = batch["slots"].float()
            last_slot = slots[:, -1] if slots.ndim == 4 else slots
        elif "pixels" in batch:
            if self.object_encoder is None:
                raise RuntimeError(
                    "This dataset contains pixels but the object encoder was not "
                    "configured. Run ./setup.sh and leave data.encoder enabled."
                )
            with torch.no_grad():
                result = self.object_encoder(
                    batch["pixels"].float(),
                    initial_slots=initial_slots,
                    return_last_slot=return_last_slot,
                )
            if return_last_slot:
                slots, last_slot = result
            else:
                slots = result
        else:
            raise KeyError("The dataset needs a 'pixels' or 'slots' column")

        expected = (self.num_object_slots, self.slot_dim)
        if tuple(slots.shape[-2:]) != expected:
            raise ValueError(
                f"Expected slots shaped (..., {expected[0]}, {expected[1]}), "
                f"got {tuple(slots.shape)}"
            )
        return (slots, last_slot) if return_last_slot else slots

    def encode(self, batch):
        slots = self.encode_objects(batch)
        batch_size, time = slots.shape[:2]
        nodes = [slots]

        if self.proprio_encoder is not None:
            proprio = batch.get("proprio")
            if proprio is None:
                proprio = slots.new_zeros(batch_size, time, self.proprio_dim)
            nodes.append(self.proprio_encoder(proprio).unsqueeze(2))

        if self.action_encoder is not None:
            action = batch.get("action")
            input_dim = self.action_dim * self.action_frameskip
            if action is None:
                action = slots.new_zeros(batch_size, time, input_dim)
            if action.shape[-1] != input_dim:
                raise ValueError(
                    f"Expected grouped actions with dimension {input_dim}; "
                    f"got {action.shape[-1]}"
                )
            nodes.append(self.action_encoder(action).unsqueeze(2))
        return torch.cat(nodes, dim=2)

    def predict(self, history, inference=False):
        return (
            self.predictor.inference(history) if inference else self.predictor(history)
        )

    def _planning_inputs(self, info):
        source = info.get("slots", info.get("pixels"))
        if source is None:
            raise KeyError("Planning requires 'pixels' or 'slots' in info")
        identity = []
        for name in ("id", "step_idx"):
            value = info.get(name)
            if torch.is_tensor(value):
                # Candidate values are repeated.  Keep a value-based token so
                # allocator reuse cannot turn the next MPC state into a false
                # cache hit.  This mirrors the released model's id/step check.
                value = value[:, 0].detach().cpu().contiguous()
                identity.append((name, tuple(value.shape), value.numpy().tobytes()))
        key = (
            tuple(identity)
            if identity
            else (
                source.data_ptr(),
                tuple(source.shape),
                source.device,
                source.dtype,
            )
        )
        if key in self._planning_cache:
            return self._planning_cache[key]

        # Candidate-expanded info is identical along dimension 1. Encode once.
        batch = {"slots" if "slots" in info else "pixels": source[:, 0]}
        object_history, previous_slots = self.encode_objects(
            batch, return_last_slot=True
        )

        goal_source = info.get("goal_slots", info.get("goal"))
        if goal_source is None:
            raise KeyError("Planning requires 'goal' or 'goal_slots' in info")
        goal_batch = {"slots" if "goal_slots" in info else "pixels": goal_source[:, 0]}
        # The released policy initializes goal slot attention from the current
        # frame's final slots, preserving object identity across current/goal.
        goal_objects = self.encode_objects(goal_batch, initial_slots=previous_slots)
        goal_objects = goal_objects[:, -1] if goal_objects.ndim == 4 else goal_objects

        goal_proprio = None
        if self.proprio_encoder is not None and "goal_proprio" in info:
            raw_goal_proprio = info["goal_proprio"][:, 0].float()
            if raw_goal_proprio.ndim == 2:
                raw_goal_proprio = raw_goal_proprio.unsqueeze(1)
            goal_proprio = self.proprio_encoder(raw_goal_proprio)[:, -1]

        self._planning_cache = {
            key: (object_history, goal_objects, goal_proprio)
        }
        return object_history, goal_objects, goal_proprio

    def _compose_planning_history(self, objects, info, num_candidates):
        batch_size, time = objects.shape[:2]
        objects = objects.unsqueeze(1).expand(
            batch_size, num_candidates, *objects.shape[1:]
        )
        nodes = [objects]

        if self.proprio_encoder is not None:
            proprio = info.get("proprio")
            if proprio is None:
                proprio_nodes = objects.new_zeros(
                    batch_size, num_candidates, time, 1, self.slot_dim
                )
            else:
                proprio = proprio[:, :, -time:]
                flat = rearrange(proprio, "b n t d -> (b n) t d")
                proprio_nodes = rearrange(
                    self.proprio_encoder(flat),
                    "(b n) t d -> b n t 1 d",
                    b=batch_size,
                    n=num_candidates,
                )
            nodes.append(proprio_nodes)

        if self.action_encoder is not None:
            nodes.append(
                objects.new_zeros(batch_size, num_candidates, time, 1, self.slot_dim)
            )
        return torch.cat(nodes, dim=3)

    def rollout(self, info, action_candidates):
        objects, _, _ = self._planning_inputs(info)
        batch_size, num_candidates = action_candidates.shape[:2]
        history = self._compose_planning_history(objects, info, num_candidates)
        history = rearrange(history, "b n t s d -> (b n) t s d")
        flat_actions = rearrange(action_candidates, "b n t d -> (b n) t d")
        observed_frames = history.shape[1]
        if flat_actions.shape[1] < observed_frames:
            raise ValueError(
                "Planning horizon must cover every observed history frame: "
                f"horizon={flat_actions.shape[1]}, observations={observed_frames}"
            )

        # As in the original policy, action 0 belongs to the observed latent.
        if self.action_encoder is not None:
            initial_actions = self.action_encoder(
                flat_actions[:, :observed_frames]
            )
            history[:, :, self.action_node] = initial_actions

        remaining_actions = flat_actions[:, observed_frames:]
        current_step = 0
        while current_step < remaining_actions.shape[1]:
            context = history[:, -self.history_frames :]
            prediction = self.predict(context, inference=True)
            prediction = self._reorder_prediction(prediction, history[:, -1])
            steps = min(
                prediction.shape[1], remaining_actions.shape[1] - current_step
            )
            prediction = prediction[:, :steps].clone()
            if self.action_encoder is not None:
                action_nodes = self.action_encoder(
                    remaining_actions[:, current_step : current_step + steps]
                )
                prediction[:, :, self.action_node] = action_nodes
            history = torch.cat([history, prediction], dim=1)
            current_step += steps

        # One final prediction represents the state reached after the last
        # injected action.  Its action node is intentionally left predicted.
        final_prediction = self.predict(
            history[:, -self.history_frames :], inference=True
        )[:, :1]
        final_prediction = self._reorder_prediction(
            final_prediction, history[:, -1]
        )
        history = torch.cat([history, final_prediction], dim=1)
        return rearrange(
            history,
            "(b n) t s d -> b n t s d",
            b=batch_size,
            n=num_candidates,
        )

    def _reorder_prediction(self, prediction, reference):
        """Hungarian-align predicted object slots and preserve AP nodes."""
        objects = prediction[:, 0, : self.num_object_slots]
        reference_objects = reference[:, : self.num_object_slots]
        pair_cost = (
            objects.unsqueeze(2) - reference_objects.unsqueeze(1)
        ).square().sum(dim=-1)
        reordered = prediction.clone()
        for batch_index, matrix in enumerate(pair_cost):
            rows, columns = linear_sum_assignment(
                matrix.detach().float().cpu().numpy()
            )
            inverse = torch.empty(
                self.num_object_slots,
                dtype=torch.long,
                device=prediction.device,
            )
            inverse[torch.as_tensor(columns, device=prediction.device)] = (
                torch.as_tensor(rows, device=prediction.device)
            )
            reordered[batch_index, :, : self.num_object_slots] = prediction[
                batch_index, :, inverse
            ]
        return reordered

    def criterion(
        self,
        predicted_objects,
        goal_objects,
        predicted_proprio=None,
        goal_proprio=None,
    ):
        pair_cost = (
            predicted_objects.unsqueeze(-2) - goal_objects.unsqueeze(-3)
        ).square().mean(dim=-1)
        costs = []
        for batch_costs in pair_cost:
            candidate_costs = []
            for matrix in batch_costs:
                rows, cols = linear_sum_assignment(
                    matrix.detach().float().cpu().numpy()
                )
                candidate_costs.append(matrix[rows, cols].mean())
            costs.append(torch.stack(candidate_costs))
        total = torch.stack(costs)
        if predicted_proprio is not None and goal_proprio is not None:
            total = total + (predicted_proprio - goal_proprio).square().mean(dim=-1)
        return total

    @torch.inference_mode()
    def get_cost(self, info, action_candidates):
        predictions = self.rollout(info, action_candidates)
        _, goal_objects, goal_proprio = self._planning_inputs(info)
        goal_objects = goal_objects.unsqueeze(1).expand(
            predictions.shape[0], predictions.shape[1], *goal_objects.shape[1:]
        )
        predicted_final = predictions[:, :, -1]
        predicted_proprio = (
            predicted_final[:, :, self.proprio_node]
            if self.proprio_node is not None and goal_proprio is not None
            else None
        )
        if goal_proprio is not None:
            goal_proprio = goal_proprio.unsqueeze(1).expand_as(predicted_proprio)
        return self.criterion(
            predicted_final[:, :, : self.num_object_slots],
            goal_objects,
            predicted_proprio,
            goal_proprio,
        )


__all__ = ["CJEPA", "MaskedSlotPredictor", "NodeEmbedder"]
