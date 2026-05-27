"""BERT-style MLM from scratch, in PyTorch primitives.

Builds the encoder block, training loop, and masking strategy directly from
nn.Linear / nn.LayerNorm — no nn.MultiheadAttention, no nn.TransformerEncoderLayer.
The goal is to see every line of the data path so the paper feels concrete.

What's faithful to the paper:
  - Multi-head scaled dot-product self-attention with the standard QKV projection.
  - Pre-norm Transformer block (LayerNorm-first; the paper used post-norm but
    pre-norm is the modern default and trains more stably at this scale).
  - 80/10/10 masking strategy on 15% of tokens.
  - MLM head with weight-tied input embeddings.

What's simplified:
  - Toy synthetic corpus (no WordPiece, no real text) so the file is self-contained.
  - No NSP. RoBERTa showed it doesn't earn its keep.
  - Tiny model (4 layers, 64 hidden, 4 heads) so it trains on CPU in under a minute.

Run:
    python bert_mlm_from_scratch.py
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------- hyperparameters ----------
VOCAB_SIZE = 64
SEQ_LEN = 16
HIDDEN = 64
N_HEADS = 4
N_LAYERS = 4
FFN_MULT = 4
DROPOUT = 0.1
MASK_PROB = 0.15
BATCH_SIZE = 32
N_STEPS = 400
LR = 5e-4

# Reserved token ids.
PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3
FIRST_REAL_TOKEN = 4


class MultiHeadSelfAttention(nn.Module):
    """The block most people gloss over — written out explicitly here."""

    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, t, c = x.shape
        # Project to Q, K, V in one matmul then split.
        qkv = self.qkv(x)  # (b, t, 3c)
        q, k, v = qkv.chunk(3, dim=-1)
        # (b, t, c) -> (b, n_heads, t, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention.
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (b, h, t, t)
        # attn_mask: (b, 1, 1, t) with 0 for real tokens, -inf for padding.
        scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v  # (b, h, t, head_dim)
        # Concat heads.
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm: x = x + drop(attn(norm(x))); x = x + drop(ffn(norm(x)))."""

    def __init__(self, hidden: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = MultiHeadSelfAttention(hidden, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MiniBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.emb_norm = nn.LayerNorm(hidden)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)

        # Tied MLM head: reuse the token embedding matrix as the output projection.
        # This is the standard BERT trick — halves the param count of the head.
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        b, t = input_ids.shape
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_norm(x))
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        # Tied projection.
        logits = x @ self.tok_emb.weight.T + self.mlm_bias  # (b, t, vocab)
        return logits


def make_synthetic_corpus(n_sentences: int = 2000) -> list[list[int]]:
    """Synthetic corpus with structure the model can actually learn.

    Each "sentence" is built from a small set of bigram patterns. The model has
    to learn that token X tends to be followed by token Y, so when we mask Y it
    should look at neighbouring X tokens to predict it.
    """
    bigram_pairs = [(i, i + 1) for i in range(FIRST_REAL_TOKEN, VOCAB_SIZE - 1, 2)]
    sentences: list[list[int]] = []
    for _ in range(n_sentences):
        length = SEQ_LEN - 2  # leave room for [CLS] and [SEP]
        body: list[int] = []
        while len(body) < length:
            a, b = random.choice(bigram_pairs)
            body.extend([a, b])
        body = body[:length]
        sentences.append([CLS_ID, *body, SEP_ID])
    return sentences


def apply_mlm_masking(
    input_ids: Tensor, mask_prob: float, vocab_size: int
) -> tuple[Tensor, Tensor]:
    """Implements the 80/10/10 strategy.

    Returns (masked_input, labels). Labels are -100 (CE ignore_index) for
    positions we do NOT predict — i.e. unmasked positions and special tokens.
    """
    labels = input_ids.clone()
    # Probability matrix; never mask special tokens.
    special = (input_ids == PAD_ID) | (input_ids == CLS_ID) | (input_ids == SEP_ID)
    probs = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
    probs.masked_fill_(special, 0.0)
    selected = torch.bernoulli(probs).bool()

    # Anywhere we DIDN'T select, ignore in loss.
    labels[~selected] = -100

    # Among selected: 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool() & selected
    # 10% -> random token (of the remaining selected positions, half)
    random_token = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & selected & ~replace_mask
    # 10% -> unchanged (the remaining selected positions)

    masked = input_ids.clone()
    masked[replace_mask] = MASK_ID
    if random_token.any():
        rand_ids = torch.randint(FIRST_REAL_TOKEN, vocab_size, (int(random_token.sum().item()),), device=input_ids.device)
        masked[random_token] = rand_ids
    return masked, labels


def build_attn_mask(input_ids: Tensor) -> Tensor:
    """(b, 1, 1, t) additive mask: 0 for real tokens, -inf for padding."""
    pad = (input_ids == PAD_ID).unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)
    return pad.float().masked_fill(pad, float("-inf"))


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    corpus = make_synthetic_corpus(n_sentences=3000)
    data = torch.tensor(corpus, dtype=torch.long, device=device)

    model = MiniBERT(VOCAB_SIZE, SEQ_LEN, HIDDEN, N_HEADS, N_LAYERS, FFN_MULT, DROPOUT).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    model.train()
    for step in range(1, N_STEPS + 1):
        idx = torch.randint(0, data.shape[0], (BATCH_SIZE,), device=device)
        batch = data[idx]
        masked, labels = apply_mlm_masking(batch, MASK_PROB, VOCAB_SIZE)
        attn_mask = build_attn_mask(masked)

        logits = model(masked, attn_mask)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                preds = logits.argmax(-1)
                mask = labels != -100
                acc = (preds[mask] == labels[mask]).float().mean().item() if mask.any() else 0.0
            print(f"step {step:4d}  loss {loss.item():.4f}  masked-token acc {acc:.3f}")

    print("\nDone. A from-scratch BERT trained on a toy corpus.")
    print("If you saw loss drop from ~4.2 -> ~1.5 and accuracy climb past 0.5, the model is learning bigram structure.")


if __name__ == "__main__":
    train()
