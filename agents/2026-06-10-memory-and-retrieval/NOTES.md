# Learning log — Agent memory and retrieval

What worked:
- Two-path consolidation was the key insight. When a turn starts with an
  explicit cue ("Remember this: …"), strip the prefix and store directly —
  no heuristic needed, importance=2.0. The heuristic path (linking-verb scan)
  catches organic declarative sentences in free conversation. Together they
  cover both "deliberate encoding" and "incidental learning", which mirrors
  how human episodic vs semantic memory work.
- TF-IDF + cosine similarity needs zero ML infrastructure and still achieves
  recall@1 = 100% on content-rich queries. The secret: each query is basically
  a restatement of the stored fact, so the token overlap is very high. TF-IDF
  gives credit to rare, informative tokens (e.g. "AdamW", "hippocampal") over
  common ones, which is exactly what you want for fact retrieval.
- JSON persistence + embedding cache is the right trade-off at this scale.
  Save the TF-IDF vectors as float lists alongside the text. On load, restore
  vectors directly — no refit needed — so the first query after reload is fast.
- hit_count × importance as a retrieval priority signal: facts that are
  retrieved repeatedly get importance-boosted. This is a cheap proxy for the
  "spacing effect" — well-rehearsed memories are stronger.

What surprised me:
- The short-term window size matters a lot for consolidation. With window=8 and
  24 turns in the planting session, only the last 4 facts survived to flush().
  The fix (window=32 for the planting session) is obvious in hindsight but easy
  to miss. Real agents should either use a bigger context window or flush
  incrementally rather than all-at-end.
- recall@1 = recall@5 = 100% when queries are near-paraphrases of stored text.
  This collapses the ranking comparison. A harder eval would paraphrase the
  queries with different vocabulary ("neural net regularisation technique
  that drops random units" instead of "Dropout regularises…") — those are the
  cases where TF-IDF breaks and dense embeddings (sentence-transformers) win.
- TF-IDF silently fails on queries with only stopwords or very short queries
  ("what is the learning rate?") because the idf vector suppresses common
  terms. The cosine of a zero vector is 0 — returning "no memory found"
  correctly rather than hallucinating. That's actually the right behaviour.

What I'd try next:
- Replace TF-IDF with sentence-transformers (or even a local Ollama embedding)
  for dense retrieval. Compare recall on paraphrase queries where TF-IDF fails.
- Implement a forgetting curve: decay importance by 0.95 per day if a record is
  not retrieved. Facts that are never re-accessed should eventually drop below a
  pruning threshold and be evicted. Tests whether the "spaced repetition" effect
  shows up in recall@k.
- Add a memory size limit (e.g. top 100 by importance) and measure recall
  degradation as the store grows. This is the practical deployment constraint —
  you can't grow a vector store unboundedly and still have fast cosine search.
- Compare against a BM25 retriever (term-frequency + IDF + document-length
  normalisation). BM25 is the standard baseline for sparse retrieval and
  typically beats vanilla TF-IDF on short queries. The implementation is 10
  extra lines.
