# Paper notes — BERT

**Paper:** Devlin et al., 2018 — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*

## Summary

BERT is a stack of Transformer **encoder** layers (no decoder) pretrained on raw text with two self-supervised objectives — **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)** — and then fine-tuned on downstream tasks by adding a small task-specific head and updating all weights. The key shift from prior models (ELMo, GPT-1) is that BERT attends to context on **both sides** of every token simultaneously, which is only possible because MLM hides the targets rather than predicting the next token autoregressively.

## Key ideas

- **Bidirectional attention.** A standard left-to-right LM can't see future tokens. BERT replaces left-to-right with full self-attention and protects the labels by masking instead.
- **MLM objective.** Randomly select 15% of WordPiece tokens. Of those: 80% replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged. Predict the original token at every selected position. The 10% + 10% noise prevents the model from learning "only look at `[MASK]` positions" — at fine-tune time there are no `[MASK]` tokens, so the pretraining/fine-tuning distribution mismatch is the main risk this trick mitigates.
- **NSP objective.** Concatenate two sentences `[CLS] A [SEP] B [SEP]` with a segment embedding `0/1`. Predict from `[CLS]` whether B follows A in the original corpus or was randomly sampled. Later work (RoBERTa) shows NSP contributes little — most of BERT's win is MLM.
- **Architecture.** Pre-norm Transformer encoder. `BERT-base` = 12 layers, hidden 768, 12 heads, 110M params. `BERT-large` = 24 layers, hidden 1024, 16 heads, 340M params.
- **Input.** Subword tokenization (WordPiece). Three summed embeddings per position: token + segment + learned position (max 512).
- **Fine-tuning recipe.** Add a single linear layer on top of `[CLS]` (classification) or per-token (NER, QA span). Update all weights with small learning rate (2e-5 to 5e-5), 2-4 epochs.

## Ablations worth remembering

- Removing NSP costs ~0.5 F1 on GLUE — small.
- MLM-only training underperforms in early steps but catches up — the bidirectional signal is what matters.
- Going from base to large gives 3-4 points on most tasks. Scale matters.
- "Feature-based" use (frozen BERT + task head) is competitive with full fine-tuning for some tasks (NER), but fine-tuning wins overall.

## Why this still matters

Every modern encoder (RoBERTa, DeBERTa, ModernBERT, sentence-transformers) is a direct descendant. If you're doing retrieval, classification, or embedding work in 2026, you're almost certainly using a BERT-family encoder under the hood. The decoder-only LLMs (GPT, Llama) took over generation but BERT-style encoders still dominate **representation** tasks because bidirectional context produces better fixed-vector embeddings than causal masking.

## Worked example — see `bert_mlm_from_scratch.py`

A minimal BERT-style encoder implemented in PyTorch from primitives (no `nn.MultiheadAttention`, no `nn.TransformerEncoderLayer`). Trains MLM on a synthetic corpus and prints the loss curve so you can see it actually learns. Run with:

```bash
pip install torch
python bert_mlm_from_scratch.py
```

## Open questions to revisit

- Why does the 80/10/10 mask noise work better than 100% `[MASK]`? Is the random-token swap mostly serving as regularization?
- How sensitive is downstream performance to position-embedding type (learned vs. sinusoidal vs. RoPE vs. ALiBi)? ModernBERT uses RoPE — measure.
- For my own retrieval use cases: when should I prefer a frozen BERT encoder + reranker over an end-to-end fine-tuned bi-encoder?

## References

- https://arxiv.org/abs/1810.04805 (original paper)
- http://jalammar.github.io/illustrated-bert/ (visual walkthrough)
- https://arxiv.org/abs/1907.11692 (RoBERTa — what BERT got wrong)
- https://arxiv.org/abs/2412.13663 (ModernBERT — what 6 years of progress looks like)
