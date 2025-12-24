"""Code borrowed from https://github.com/karpathy/minGPT"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, d_model, n_head, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head

        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(d_model, d_model)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, -1),
        )

    def forward(self, x):
        B, T, C = x.size()  # (B, T, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head,
                             C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head,
                               C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head,
                               C // self.n_head).transpose(1, 2)

        # causal self-attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalTransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, d_model, n_head, max_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(
        self,
        vocab_size,
        d_model,
        n_head,
        max_len,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.drop = nn.Dropout(dropout)
        # transformer
        self.blocks = nn.Sequential(*[
            CausalTransformerBlock(d_model, n_head, max_len, dropout)
            for _ in range(num_layers)
        ])
        # decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx):
        """Forward pass.

        Args:
            idx (torch.LongTensor): input token indices of shape (B, t2)
        """
        T = idx.shape[1]
        assert T <= self.max_len

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # [B, T, C]
        position_embeddings = self.pos_emb[:, :T, :]  # [1, T, C]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]

        return logits

    def generate(self, x, steps, temperature=1.0, sample=False):
        """Generate a sequence of length `steps` in an autoregressive manner.

        Args:
            x (torch.LongTensor): input token indices of shape (B, T)

        Returns:
            torch.LongTensor: generated token indices of shape (B, T + steps)
        """
        for _ in range(steps):
            x_cond = x if x.size(1) <= self.max_len else x[:, -self.max_len:]
            logits = self.forward(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        return x
