import torch
import torch.nn.functional as F

# from torchvision import transforms
import torchvision.transforms.v2 as transforms
from einops import rearrange, repeat
from torch import distributed as dist
from torch import nn


class CausalWM_Savi(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        proprio_encoder,
        decoder=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
        device="cpu",
        model_name='SAVi' # or 'StoSAVi'
    ):
        super().__init__()

        self.encoder = encoder # includes backbone and output transform
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        self.decoder = decoder
        self.history_size = history_size
        self.num_pred = num_pred
        self.device = device
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.model_name=model_name

        decoder_scale = 16  # from vqvae
        num_side_patches = 224 // decoder_scale
        self.encoder_image_size = 64
        self.encoder_transform = transforms.Compose([transforms.Resize(self.encoder_image_size)])

    def encode(
        self,
        info,
        pixels_key="pixels",
        prefix=None,
        target="embed",
        proprio_key=None,
        action_key=None,
        return_last_slot=False
    ):
        assert target not in info, f"{target} key already in info_dict"
        emb_keys = [proprio_key, action_key]
        prefix = prefix or ""
        image_size=64

        # == pixels embeddings
        pixels = info[pixels_key].float()  # (B, T, 3, H, W)
        B = pixels.shape[0]
        if pixels.shape[-2] != image_size or pixels.shape[-1] != image_size:
            transform = transforms.Compose([transforms.Resize(image_size)])
            pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
            pixels = transform(pixels)
            pixels = rearrange(pixels, "(b t) c h w -> b t c h w", b=B)
        info[pixels_key] = pixels  # store resized pixels

        if "prev_slot" in info:
            pixels_embed = self.encoder._modules["backbone"]._forward(pixels, prev_slots=info["prev_slot"])
        else:
            in_dict = {'img': pixels}  # resize for encoder
            pixels_embed = self.encoder(in_dict)
        pixels_embed = pixels_embed['post_slots'] # bs x nstep x numslot x slotfeat (8x4x7x64)
        if len(pixels_embed.shape) == 3:
            pixels_embed = pixels_embed.unsqueeze(0)  # (B, T, s, d) # batch 가 없어진건지.. temporal이 없어진건지...
        if return_last_slot:
            if "prev_slot" not in info:
                info['prev_slot'] = pixels_embed[:, -1].detach().clone()
            else:
                info.update({'prev_slot': pixels_embed[:, -1].detach().clone()})

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f"pixels_{target}"] = pixels_embed

        for key in [proprio_key, action_key]: # always in this order
            if key == "proprio":
                extr_enc = self.proprio_encoder
            elif key == "action":
                extr_enc = self.action_encoder
            else:
                continue
            extra_input = info[f"{prefix}{key}"].float()  # (B, T, dim)
            extra_embed = extr_enc(extra_input)  # (B, T, dim) -> (B, T, emb_dim)
            info[f"{key}_{target}"] = extra_embed

            # copy extra embedding across patches for each time step
            extra_tiled = repeat(extra_embed.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)

            # concatenate along feature dimension
            embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)

        return info

    def predict(self, embedding, use_inference_function: bool=False):
        """predict next latent state
        Args:
            embedding: (B, T, P, d) - P can be num_patches or num_slots
        Returns:
            preds: (B, T, P, d) for old-style or (B, num_pred, P, d) for MaskedSlotPredictor
            mask_info: (mask_indices, T) for selective loss computation
        """
        if use_inference_function: # no mask
            preds = self.predictor.inference(embedding)
            return preds
        else:
            preds, mask_indices = self.predictor(embedding)
            return preds, mask_indices

        # Output: (B, num_pred, S, 64) or (B, T+num_pred, S, 64)

    def decode(self, info):
        assert "pixels_embed" in info, "pixels_embed not in info_dict"
        pixels_embed = info["pixels_embed"]
        num_frames = pixels_embed.shape[1]

        pixels, diff = self.decoder(pixels_embed)  # (b*num_frames, 3, 224, 224)
        pixels = rearrange(pixels, "(b t) c h w -> b t c h w", t=num_frames)

        info["reconstructed_pixels"] = pixels
        info["reconstruction_diff"] = diff

        return info

    def split_embedding(self, embedding, extra_dims):  # action_dim, proprio_dim):
        split_embed = {}
        pixel_dim = embedding.shape[-1] - sum(extra_dims)

        # == pixels embedding
        split_embed["pixels_embed"] = embedding[..., :pixel_dim]

        # == extra embeddings
        start_dim = pixel_dim
        for i, key in enumerate(["proprio", "action"]):
            dim = extra_dims[i]
            extra_emb = embedding[..., start_dim : start_dim + dim]
            split_embed[f"{key}_embed"] = extra_emb[:, :, :, 0]  # all patches are the same
            start_dim += dim

        return split_embed

    def replace_action_in_embedding(self, embedding, act):
        """Replace the action embeddings in the latent state z with the provided actions.
        
        Args:
            embedding: (B, N, T, P, d) - 5D tensor with action at the end of d
            act: (B, N, T, action_dim) - raw actions to inject
        Returns:
            new_embedding: (B, N, T, P, d)
        """
        assert self.action_encoder is not None, "No action encoder defined in the model."
        n_patches = embedding.shape[3]
        B, N = act.shape[:2]
        act_flat = rearrange(act, "b n ... -> (b n) ...")
        z_act = self.action_encoder(act_flat)  # (B*N, T, A_emb)
        action_dim = z_act.shape[-1]
        act_tiled = repeat(z_act.unsqueeze(2), "(b n) t 1 a -> b n t p a", b=B, n=N, p=n_patches)
        # z (B, N, T, P, d) with d = dim + extra_dims
        # determine where action starts in the embedding
        extra_dim = sum(encoder.emb_dim for encoder in [self.proprio_encoder, self.action_encoder])
        pixel_dim = embedding.shape[-1] - extra_dim

        start = pixel_dim + self.proprio_encoder.emb_dim
        # for key, encoder in self.extra_encoders.items():
        #     if key == "action":
        #         break
        #     start += encoder.emb_dim

        prefix = embedding[..., :start]
        suffix = embedding[..., start + action_dim :]

        new_embedding = torch.cat([prefix, act_tiled, suffix], dim=-1)

        # embedding[..., start : start + action_dim] = act_tiled
        # return embedding
        return new_embedding

    def _replace_action_flat(self, embedding, actions):
        """Replace actions in embedding for flattened batch (no N dimension).
        
        Args:
            embedding: (B, T, S, D) - 4D tensor with action at the end of D
            actions: (B, T, action_dim) - raw actions to inject
        Returns:
            new_embedding: (B, T, S, D)
        """
        assert self.action_encoder is not None, "No action encoder defined in the model."
        n_patches = embedding.shape[2]
        act_embed = self.action_encoder(actions)  # (B, T, action_embed_dim)
        action_dim = act_embed.shape[-1]
        act_tiled = repeat(act_embed.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
        new_embedding = torch.cat([embedding[..., :-action_dim], act_tiled], dim=-1)
        return new_embedding

    def rollout(self, info, action_sequence):
        """Rollout the world model given an initial observation and a sequence of actions.
        
        Non-autoregressive rollout for MaskedSlotPredictor:
        - Predicts pred_frames at a time (e.g., 3 frames per call)
        - If horizon > pred_frames, reconstructs history using predicted frames
        - Repeats until all required future steps are predicted

        Params:
            info: dict containing 'pixels', 'proprio', etc.
                  pixels shape: (B, N, T_history, C, H, W) where N = num_candidates
                  - T_history comes from world.history_size
            action_sequence: (B, N, horizon, action_dim * action_block)
                  - This contains ONLY future actions (no history)
                  - horizon = plan_config.horizon (number of planning steps)
                  - Each action covers action_block environment steps

        Returns:
            info: dict with 'predicted_embedding' of shape (B, N, T_history + horizon, S, D)
        """

        assert "pixels" in info, "pixels not in info_dict"
        proprio_key = "proprio" if "proprio" in info else None
        emb_keys = [proprio_key] if proprio_key in info else []
        
        # == add action to info dict

        n_obs = info["pixels"].shape[2]
        act_0 = action_sequence[:, :, :n_obs]
        info["action"] = act_0
        # check if we have already computed the initial embedding for this state
        if (
            hasattr(self, "_init_cached_info")
            and torch.equal(self._init_cached_info["id"], info["id"][:, 0])
            and torch.equal(self._init_cached_info["step_idx"], info["step_idx"][:, 0])
        ):
            init_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in self._init_cached_info.items()}
        else:
            # prepare init_info_dict
            init_info_dict = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    init_info_dict[k] = info[k][:, 0]  # (B, T_history, ...)

            init_info_dict = self.encode(
                init_info_dict,
                pixels_key="pixels",
                target="embed",
                proprio_key=proprio_key,
                action_key="action",
                return_last_slot=True
            )
            # repeat copy for each action candidate
            init_info_dict["embed"] = (
                init_info_dict["embed"]
                .unsqueeze(1)
                .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict["embed"].ndim - 1)))
                .clone()
            )
            init_info_dict["pixels_embed"] = (
                init_info_dict["pixels_embed"]
                .unsqueeze(1)
                .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict["pixels_embed"].ndim - 1)))
            )

            for key in emb_keys:
                init_info_dict[f"{key}_embed"] = (
                    init_info_dict[f"{key}_embed"]
                    .unsqueeze(1)
                    .expand(-1, action_sequence.shape[1], *([-1] * (init_info_dict[f"{key}_embed"].ndim - 1)))
                )

            init_info_dict = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in init_info_dict.items()}
            self._init_cached_info = init_info_dict

        # Get encoded history
        info["embed"] = init_info_dict["embed"]
        info["pixels_embed"] = init_info_dict["pixels_embed"]
        info["prev_slot"] = init_info_dict["prev_slot"]
        for key in emb_keys:
            info[f"{key}_embed"] = init_info_dict[f"{key}_embed"]

        # actually compute the embedding of action for each candidate
        info["embed"] = self.replace_action_in_embedding(info["embed"], action_sequence[:, :, :n_obs])
        # action_dim = init_info_dict["action_embed"].shape[-1]
        info["action_embed"] = action_sequence[:, :, :n_obs]  # info["embed"][:, :, :n_obs, 0, -action_dim:]

        # number of step to predict
        act_pred = action_sequence[:, :, n_obs:]
        n_steps = act_pred.shape[2]

        # == initial embedding
        z = info["embed"]
        B, N = z.shape[:2]


        # we flatten B and N to process all candidates in a single batch in the predictor
        z_flat = rearrange(z, "b n ... -> (b n) ...").clone()
        act_pred_flat = rearrange(act_pred, "b n ... -> (b n) ...")


        # Collect all predictions
        current_step = 0

        while current_step < n_steps:
            # Get current history window (last history_size frames)
            history_input = z_flat[:, -self.history_size:]  # (B*N, history_size, S, D)

            # Predict pred_frames future frames at once
            pred_embed = self.predict(history_input, use_inference_function=True)  # (B*N, pred_frames, S, D)
            pred_frames = pred_embed.shape[1]

            # How many steps to use from this prediction
            steps_this_round = min(pred_frames, n_steps - current_step)

            # Inject actions into predicted frames
            new_action = act_pred_flat[None, :, current_step:current_step + steps_this_round, :]  # (B*N, steps_this_round, action_dim)
            
            # Only replace action in the frames we'll actually use
            pred_to_use = pred_embed[:, :steps_this_round].clone()  # (B*N, steps_this_round, S, D)
            new_embed = self.replace_action_in_embedding(pred_to_use.unsqueeze(0), new_action)[0]

            # Update z_flat by appending predictions (for next iteration's history)
            z_flat = torch.cat([z_flat, new_embed], dim=1)

            current_step += steps_this_round

        # Predict one more step without action (final state after all actions)
        # This matches the original behavior of predicting n+t+1 states for n+t actions
        final_history = z_flat[:, -self.history_size:]
        final_pred = self.predict(final_history, use_inference_function=True)[:, :1]  # Just first predicted frame
        z_flat = torch.cat([z_flat, final_pred], dim=1)

        # Reshape back to (B, N, T_total, S, D)
        z = rearrange(z_flat, "(b n) ... -> b n ...", b=B, n=N)

        # == update info dict with predicted embeddings
        info["predicted_embedding"] = z

        # Extract embedding components for cost computation
        extra_dims = []
        # proprio first
        if self.proprio_encoder is not None:
            extra_dims.append(info["proprio_embed"].shape[-1])
        if self.action_encoder is not None:
            extra_dims.append(self.action_encoder.emb_dim)

        splitted_embed = self.split_embedding(z, extra_dims)
        info.update({f"predicted_{k}": v for k, v in splitted_embed.items()})

        return info

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost for planning. Should be overridden for custom costs."""
        proprio_key = "proprio" if "proprio" in info_dict else None
        emb_keys = [proprio_key] if proprio_key in info_dict else [] 
        cost = 0.0

        for key in emb_keys + ["pixels"]:
            preds = info_dict[f"predicted_{key}_embed"]
            goal = info_dict[f"{key}_goal_embed"]
            cost = cost + F.mse_loss(preds[:, :, -1:], goal[:, :, -1:], reduction="none").mean(dim=tuple(range(2, preds.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "action" in info_dict, "action key must be in info_dict"
        assert "pixels" in info_dict, "pixels key must be in info_dict"

        # move to device and unsqueeze time
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(next(self.parameters()).device)
        proprio_key = "proprio" if "proprio" in info_dict else None
        emb_keys = [proprio_key] if proprio_key in info_dict else []
        # == run world model
        info_dict = self.rollout(info_dict, action_candidates)
        # == get the goal embedding

        # check if we have already computed the goal embedding for this goal
        if (
            hasattr(self, "_goal_cached_info")
            and torch.equal(self._goal_cached_info["id"], info_dict["id"][:, 0])
            and torch.equal(self._goal_cached_info["step_idx"], info_dict["step_idx"][:, 0])
        ):
            goal_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in self._goal_cached_info.items()}

        else:
            # prepare goal_info_dict
            goal_info_dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    # prev_slot has shape [B, num_slots, slot_size] without candidate dim
                    # so we should not index it with [:, 0]
                    if k == "prev_slot":
                        goal_info_dict[k] = info_dict[k]  # keep as is: [B, num_slots, slot_size]
                    else:
                        goal_info_dict[k] = info_dict[k][:, 0]  # (B, ...)
            assert "prev_slot" in info_dict, "prev_slot must be in info_dict for goal encoding"
            goal_info_dict = self.encode(
                goal_info_dict,
                target="goal_embed",
                pixels_key="goal",
                prefix="goal_",
                proprio_key=proprio_key,
                action_key=None,
            )

            goal_info_dict["goal_embed"] = (
                goal_info_dict["goal_embed"]
                .unsqueeze(1)
                .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict["goal_embed"].ndim - 1)))
            )

            goal_info_dict["pixels_goal_embed"] = (
                goal_info_dict["pixels_goal_embed"]
                .unsqueeze(1)
                .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict["pixels_goal_embed"].ndim - 1)))
            )

            for key in emb_keys:
                goal_info_dict[f"{key}_goal_embed"] = (
                    goal_info_dict[f"{key}_goal_embed"]
                    .unsqueeze(1)
                    .expand(-1, action_candidates.shape[1], *([-1] * (goal_info_dict[f"{key}_goal_embed"].ndim - 1)))
                )

            goal_info_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in goal_info_dict.items()}
            self._goal_cached_info = goal_info_dict

        info_dict["goal_embed"] = goal_info_dict["goal_embed"]
        info_dict["pixels_goal_embed"] = goal_info_dict["pixels_goal_embed"]

        for key in emb_keys:
            info_dict[f"{key}_goal_embed"] = goal_info_dict[f"{key}_goal_embed"]



        # cost = 0.0

        # for key in emb_keys + ["pixels"]:
        #     preds = info_dict[f"predicted_{key}_embed"]
        #     goal = info_dict[f"{key}_goal_embed"]
        #     cost = cost + F.mse_loss(preds[:, :, -1:], goal, reduction="none").mean(dim=tuple(range(2, preds.ndim)))

        return self.criterion(info_dict, action_candidates)


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, x):
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class CausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim))  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, num_frames)
        self.pool = pool

    def forward(self, x):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

        self.register_buffer("bias", self.generate_mask_matrix(num_patches, num_frames))

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        attn_mask = self.bias[:, :, :T, :T] == 1  # bool mask

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        rows = []
        for i in range(nwindow):
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
