"""ALOE transformer used by the original C-JEPA CLEVRER experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def append_token(values: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    token = token.to(dtype=values.dtype).view(*([1] * (values.ndim - 1)), -1)
    return torch.cat([values, token.expand(*values.shape[:-1], -1)], dim=-1)


class ALOE(nn.Module):
    """Slot-based ALOE model for descriptive and multiple-choice questions.

    The dimensions and token construction follow the ALOE implementation that
    was previously vendored through SlotFormer. Only its NeRV transformer
    wrapper is replaced with the equivalent native PyTorch encoder.
    """

    def __init__(
        self,
        *,
        num_slots: int = 7,
        slot_dim: int = 128,
        sample_frames: int = 25,
        max_question_len: int = 20,
        max_choice_len: int = 12,
        question_vocab_size: int = 82,
        answer_vocab_size: int = 22,
        input_dim: int = 16,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 512,
        mlp_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        language_dim = input_dim - 2
        token_dim = input_dim + 2
        model_dim = token_dim * num_heads

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sample_frames = sample_frames
        self.max_question_len = max_question_len
        self.max_choice_len = max_choice_len
        self.sequence_len = (
            1
            + num_slots * sample_frames
            + max_question_len
            + max_choice_len
        )

        self.register_buffer("text_token", torch.tensor([1.0, 0.0]))
        self.register_buffer("vision_token", torch.tensor([0.0, 1.0]))
        self.register_buffer("cls_token", torch.tensor([0.0, 1.0]))
        self.register_buffer("mc_question_token", torch.tensor([1.0, 0.0]))
        self.register_buffer("mc_choice_token", torch.tensor([0.0, 1.0]))

        self.question_embedding = nn.Embedding(question_vocab_size, language_dim)
        self.question_projection = nn.Linear(token_dim, model_dim)
        self.vision_projection = nn.Linear(slot_dim + 2, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, self.sequence_len, model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.descriptive_head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, answer_vocab_size),
        )
        self.multiple_choice_head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def _encode(
        self,
        slots: torch.Tensor,
        language: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = language.shape[0]
        vision = append_token(slots.flatten(1, 2), self.vision_token)
        vision = self.vision_projection(vision)
        language = append_token(language, self.text_token)
        language = self.question_projection(language)
        values = torch.cat([self.cls.expand(batch_size, -1, -1), vision, language], 1)
        if values.shape[1] != self.sequence_len:
            raise ValueError(
                f"ALOE expected {self.sequence_len} tokens, got {values.shape[1]}"
            )
        no_padding = torch.zeros(
            batch_size,
            1 + vision.shape[1],
            dtype=torch.bool,
            device=padding.device,
        )
        padding = torch.cat([no_padding, padding.bool()], dim=1)
        return self.transformer(values + self.position, src_key_padding_mask=padding)

    def _descriptive(self, batch: dict[str, torch.Tensor]):
        if batch["cls_q_tokens"].shape[0] == 0:
            return None
        language = self.question_embedding(batch["cls_q_tokens"].long())
        language = append_token(language, self.cls_token)
        encoded = self._encode(
            batch["cls_video_emb"], language, batch["cls_q_pad_mask"]
        )
        return self.descriptive_head(encoded[:, 0])

    def _multiple_choice(self, batch: dict[str, torch.Tensor]):
        if batch["mc_q_tokens"].shape[0] == 0:
            return None
        slots = batch["mc_video_emb"][batch["mc_flag"].long()]
        language = self.question_embedding(batch["mc_q_tokens"].long())
        question = append_token(
            language[:, : self.max_question_len], self.mc_question_token
        )
        choice = append_token(
            language[:, self.max_question_len :], self.mc_choice_token
        )
        encoded = self._encode(
            slots,
            torch.cat([question, choice], dim=1),
            batch["mc_q_pad_mask"],
        )
        return self.multiple_choice_head(encoded[:, 0]).flatten()

    def forward(self, batch: dict[str, torch.Tensor]):
        return {
            "cls_answer_logits": self._descriptive(batch),
            "mc_answer_logits": self._multiple_choice(batch),
        }

    @staticmethod
    def loss(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]):
        cls_logits = output["cls_answer_logits"]
        mc_logits = output["mc_answer_logits"]
        present = next(value for value in output.values() if value is not None)
        zero = present.new_zeros(())
        cls_loss = (
            F.cross_entropy(cls_logits, batch["cls_label"].long())
            if cls_logits is not None
            else zero
        )
        mc_loss = (
            F.binary_cross_entropy_with_logits(
                mc_logits, batch["mc_label"].to(dtype=mc_logits.dtype)
            )
            if mc_logits is not None
            else zero
        )
        return {"loss": cls_loss + mc_loss, "cls_loss": cls_loss, "mc_loss": mc_loss}
