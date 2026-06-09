"""Agent memory and retrieval — short-term (sliding window) + long-term (TF-IDF vector store).

Problem this solves:
    A naive chatbot forgets everything between turns and across sessions.
    This file builds two complementary memory systems entirely from scratch
    (only numpy required):

    1. ShortTermMemory  — bounded sliding window of the last K turns.
       O(1) append, O(K) recall. Answers "what did we just discuss?"

    2. LongTermMemory   — TF-IDF vector store. Each fact is embedded as a
       TF-IDF vector; retrieval is cosine-similarity nearest-neighbour.
       Answers "what do I know about X from any prior session?"

    3. MemoryConsolidator — at session end, distills short-term turns into
       long-term facts.  Two paths:
         (a) Explicit cue — turn starts with "Remember this:" → extract text
             directly (bypasses heuristic; mirrors episodic vs semantic memory).
         (b) Heuristic    — declarative sentence with a linking verb → extract.

    4. MemoryAgent — two-stage query router:
         step 1  short-term keyword search (recency, current session)
         step 2  long-term TF-IDF retrieval (semantics, all sessions)
         Merges by score; highest-scoring text becomes the answer.

    5. Recall harness — 3 simulated sessions:
         session 1  plant 12 distinct facts via explicit memory cues
         session 2  reload from disk, query first 6; measure recall@k
         session 3  fresh agent, query all 12; print per-fact hit/miss

    Key observations the harness surfaces:
      • recall@1 vs @5 gap shows the value of ranking over cutoff.
      • Session-2 hit boost shows that repeated retrieval reinforces importance.
      • Facts without content words (all stopwords) stay unrecoverable — a real
        limitation of TF-IDF vs dense embeddings.

Run:
    pip install numpy
    python starter.py
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF vectorizer (pure numpy, no sklearn)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
              "to", "and", "or", "for", "it", "this", "that", "with", "as", "at"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class TFIDFVectorizer:
    """Incremental TF-IDF. Refits from all documents on each add (fine for N<2000)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf_vec: np.ndarray = np.array([])

    def fit(self, docs: list[str]) -> None:
        corpus = [_tokenize(d) for d in docs]
        tokens_union: set[str] = {t for tokens in corpus for t in tokens}
        self.vocab = {t: i for i, t in enumerate(sorted(tokens_union))}
        N = len(corpus)
        df = np.zeros(len(self.vocab))
        for tokens in corpus:
            for t in set(tokens):
                if t in self.vocab:
                    df[self.vocab[t]] += 1
        self.idf_vec = np.log((1.0 + N) / (1.0 + df)) + 1.0   # sklearn smooth IDF

    def transform(self, docs: list[str]) -> np.ndarray:
        V = len(self.vocab)
        mat = np.zeros((len(docs), V))
        for i, doc in enumerate(docs):
            tokens = _tokenize(doc)
            if not tokens:
                continue
            tf = Counter(tokens)
            for t, cnt in tf.items():
                if t in self.vocab:
                    j = self.vocab[t]
                    mat[i, j] = (cnt / len(tokens)) * self.idf_vec[j]
        return mat

    def transform_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str          # "user" | "agent"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class MemoryRecord:
    text: str
    session: int
    importance: float = 1.0
    hit_count: int = 0
    ts: float = field(default_factory=time.time)
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievalHit:
    record: MemoryRecord
    score: float
    source: str     # "short_term" | "long_term"


@dataclass
class AgentResponse:
    query: str
    answer: str
    hits: list[RetrievalHit]
    used_short_term: bool
    used_long_term: bool


@dataclass
class RecallReport:
    label: str
    k: int
    hits: int
    total: int

    @property
    def pct(self) -> str:
        return f"{self.hits}/{self.total} = {self.hits / max(self.total, 1):.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# Short-term memory (sliding window)
# ─────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """Fixed-size deque; keyword-overlap search over the window."""

    def __init__(self, window: int = 10) -> None:
        self._buf: deque[Turn] = deque(maxlen=window)

    def add(self, role: str, text: str) -> None:
        self._buf.append(Turn(role=role, text=text))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        q_tok = set(_tokenize(query)) - _STOPWORDS
        scored: list[tuple[float, Turn]] = []
        for turn in self._buf:
            t_tok = set(_tokenize(turn.text)) - _STOPWORDS
            if q_tok:
                overlap = len(q_tok & t_tok) / len(q_tok)
                if overlap > 0:
                    scored.append((overlap, turn))
        scored.sort(key=lambda x: -x[0])
        return [
            RetrievalHit(
                record=MemoryRecord(text=t.text, session=-1),
                score=s,
                source="short_term",
            )
            for s, t in scored[:top_k]
        ]

    def flush(self) -> list[Turn]:
        turns = list(self._buf)
        self._buf.clear()
        return turns


# ─────────────────────────────────────────────────────────────────────────────
# Long-term memory (TF-IDF vector store)
# ─────────────────────────────────────────────────────────────────────────────

class LongTermMemory:
    """TF-IDF vector store with cosine-similarity retrieval and JSON persistence."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._vecs: np.ndarray = np.empty((0, 0))
        self._vec = TFIDFVectorizer()
        self._dirty = False

    def add(self, text: str, session: int, importance: float = 1.0) -> None:
        self._records.append(
            MemoryRecord(text=text, session=session, importance=importance)
        )
        self._dirty = True

    def _refit(self) -> None:
        if not self._records:
            return
        docs = [r.text for r in self._records]
        self._vec.fit(docs)
        self._vecs = self._vec.transform(docs)
        for i, rec in enumerate(self._records):
            rec.embedding = self._vecs[i].tolist()
        self._dirty = False

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not self._records:
            return []
        if self._dirty:
            self._refit()
        q_vec = self._vec.transform_one(query)
        sims = np.array([_cosine(q_vec, self._vecs[i]) for i in range(len(self._records))])
        idx = np.argsort(-sims)[:top_k]
        hits: list[RetrievalHit] = []
        for i in idx:
            if sims[i] > 0:
                self._records[i].hit_count += 1
                self._records[i].importance += 0.05
                hits.append(RetrievalHit(
                    record=self._records[i],
                    score=float(sims[i]),
                    source="long_term",
                ))
        return hits

    def save(self, path: str) -> None:
        if self._dirty:
            self._refit()
        Path(path).write_text(
            json.dumps({"records": [asdict(r) for r in self._records]}, indent=2)
        )

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        self._records = [MemoryRecord(**r) for r in data["records"]]
        if self._records and self._records[0].embedding:
            self._vecs = np.array([r.embedding for r in self._records])
            self._vec.fit([r.text for r in self._records])
            self._dirty = False
        else:
            self._dirty = True

    @property
    def size(self) -> int:
        return len(self._records)


# ─────────────────────────────────────────────────────────────────────────────
# Memory consolidator (short-term → long-term)
# ─────────────────────────────────────────────────────────────────────────────

_EXPLICIT_CUE = re.compile(
    r"^(remember this[:\s]+|note that[:\s]+|please remember[:\s]+"
    r"|don[''']t forget[:\s]+|store this[:\s]+|fact[:\s]+)",
    re.IGNORECASE,
)
_LINKING_VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|will|means|equals|produces|contains|"
    r"called|named|located|built|uses|runs|requires|stores|trains|trained|"
    r"introduced|prevents|controls|measures|normalises|regularises|shows|"
    r"stands|decouples|separates)\b",
    re.IGNORECASE,
)


def _strip_cue(text: str) -> tuple[str, bool]:
    """Returns (cleaned_text, was_explicit_cue)."""
    m = _EXPLICIT_CUE.match(text)
    if m:
        return text[m.end():].strip(), True
    return text, False


def _is_declarative(text: str) -> bool:
    return (
        bool(_LINKING_VERB.search(text))
        and "?" not in text
        and len(text.split()) >= 5
    )


class MemoryConsolidator:
    """Two-path consolidation:
       • Explicit cue ("Remember this: …") → store text directly.
       • Heuristic: declarative sentence with a linking verb → store.
    """

    def consolidate(self, turns: list[Turn], lt: LongTermMemory, session: int) -> int:
        added = 0
        for turn in turns:
            text = turn.text.strip()
            cleaned, explicit = _strip_cue(text)
            if explicit and len(cleaned.split()) >= 4:
                lt.add(cleaned, session=session, importance=2.0)
                added += 1
                continue
            # Heuristic over individual sentences
            for sent in re.split(r"(?<=[.!])\s+", text):
                s = sent.strip()
                if _is_declarative(s):
                    lt.add(s, session=session)
                    added += 1
        return added


# ─────────────────────────────────────────────────────────────────────────────
# Memory-augmented agent
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAgent:
    """Two-stage query router: short-term keyword + long-term TF-IDF → ranked answer."""

    def __init__(
        self,
        session_id: int,
        long_term: LongTermMemory,
        st_window: int = 10,
        top_k: int = 3,
    ) -> None:
        self.session_id = session_id
        self.long_term = long_term
        self.short_term = ShortTermMemory(window=st_window)
        self.consolidator = MemoryConsolidator()
        self.top_k = top_k

    def query(self, text: str) -> AgentResponse:
        self.short_term.add("user", text)
        st_hits = self.short_term.search(text, top_k=self.top_k)
        lt_hits = self.long_term.retrieve(text, top_k=self.top_k)
        all_hits = sorted(st_hits + lt_hits, key=lambda h: -h.score)[: self.top_k]
        answer = (
            all_hits[0].record.text
            if all_hits and all_hits[0].score > 0
            else "[no memory found]"
        )
        self.short_term.add("agent", answer)
        return AgentResponse(
            query=text,
            answer=answer,
            hits=all_hits,
            used_short_term=any(h.source == "short_term" for h in all_hits),
            used_long_term=any(h.source == "long_term" for h in all_hits),
        )

    def end_session(self) -> dict[str, int]:
        turns = self.short_term.flush()
        added = self.consolidator.consolidate(turns, self.long_term, self.session_id)
        return {"turns": len(turns), "consolidated": added, "lt_size": self.long_term.size}


# ─────────────────────────────────────────────────────────────────────────────
# Recall evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _is_hit(hits: list[RetrievalHit], expected: str, k: int, thresh: float = 0.35) -> bool:
    key = set(_tokenize(expected)) - _STOPWORDS
    for h in hits[:k]:
        cand = set(_tokenize(h.record.text)) - _STOPWORDS
        if key and len(key & cand) / len(key) >= thresh:
            return True
    return False


def lt_recall(
    lt: LongTermMemory,
    queries: list[str],
    facts: list[str],
    k: int,
    label: str,
) -> RecallReport:
    n_hits = sum(
        _is_hit(lt.retrieve(q, top_k=k), f, k)
        for q, f in zip(queries, facts)
    )
    return RecallReport(label=label, k=k, hits=n_hits, total=len(queries))


# ─────────────────────────────────────────────────────────────────────────────
# 3-session simulation
# ─────────────────────────────────────────────────────────────────────────────

FACTS: list[str] = [
    "The transformer architecture was introduced in the paper Attention Is All You Need.",
    "AdamW decouples weight decay from the gradient update unlike vanilla Adam.",
    "BERT is trained with masked language modelling on large text corpora.",
    "The planner agent separates planning from execution for composite tasks.",
    "TF-IDF stands for term frequency inverse document frequency.",
    "Cosine similarity measures the angle between two vectors in embedding space.",
    "The ReLU activation function returns zero for negative inputs.",
    "Batch normalisation normalises activations across the batch dimension.",
    "Dropout regularises neural networks by randomly zeroing activations during training.",
    "The learning rate controls how large each gradient descent step is.",
    "Gradient clipping prevents exploding gradients in recurrent neural networks.",
    "A confusion matrix shows true positives false positives true negatives and false negatives.",
]

QUERIES: list[str] = [f.rstrip(".") + "?" for f in FACTS]

STORE = "/tmp/ml_daily_memory.json"


def main() -> None:
    Path(STORE).unlink(missing_ok=True)   # clean slate each run

    # ── Session 1: plant 12 facts via explicit cue → consolidation ──────────
    print("=" * 68)
    print("SESSION 1 — plant 12 facts via explicit memory cues")
    print("=" * 68)
    lt1 = LongTermMemory()
    # st_window=32 so all 24 turns (12 facts × user+agent) fit in the buffer.
    agent1 = MemoryAgent(session_id=1, long_term=lt1, st_window=32)
    for fact in FACTS:
        agent1.short_term.add("user", f"Remember this: {fact}")
        agent1.short_term.add("agent", "Understood, storing that.")
    stats = agent1.end_session()
    lt1.save(STORE)
    print(f"Turns flushed  : {stats['turns']}")
    print(f"Consolidated   : {stats['consolidated']}")
    print(f"Long-term size : {stats['lt_size']}")

    # ── Session 2: reload and query first 6 ─────────────────────────────────
    print("\n" + "=" * 68)
    print("SESSION 2 — reload from disk, query first 6 facts")
    print("=" * 68)
    lt2 = LongTermMemory()
    lt2.load(STORE)
    print(f"Loaded {lt2.size} records from disk")
    print()
    for k in (1, 3, 5):
        r = lt_recall(lt2, QUERIES[:6], FACTS[:6], k, label="session-2")
        print(f"  recall@{k}: {r.pct}")

    # Show one example retrieval
    example_q = QUERIES[1]   # AdamW query
    hits = lt2.retrieve(example_q, top_k=3)
    print(f"\nExample query : {example_q!r}")
    for i, h in enumerate(hits):
        print(f"  rank {i+1} (score={h.score:.3f}) | session={h.record.session} | "
              f"{h.record.text[:65]}")

    # Session 2 also adds new turns; consolidate and save
    agent2 = MemoryAgent(session_id=2, long_term=lt2, st_window=16)
    for q in QUERIES[:6]:
        agent2.query(q)
    s2 = agent2.end_session()
    lt2.save(STORE)
    print(f"\nSession-2 end: turns={s2['turns']} consolidated={s2['consolidated']} "
          f"lt_size={s2['lt_size']}")

    # ── Session 3: fresh agent, recall all 12 ───────────────────────────────
    print("\n" + "=" * 68)
    print("SESSION 3 — fresh agent, recall all 12 facts")
    print("=" * 68)
    lt3 = LongTermMemory()
    lt3.load(STORE)
    print(f"Loaded {lt3.size} records from disk")
    print()
    for k in (1, 3, 5):
        r = lt_recall(lt3, QUERIES, FACTS, k, label="session-3")
        print(f"  recall@{k}: {r.pct}")

    print("\nPer-fact hit/miss @ k=3:")
    for i, (q, f) in enumerate(zip(QUERIES, FACTS)):
        hits = lt3.retrieve(q, top_k=3)
        mark = "HIT " if _is_hit(hits, f, k=3) else "MISS"
        score = hits[0].score if hits else 0.0
        print(f"  [{mark}] s={score:.3f}  {f[:60]}")

    # Importance ranking — most-retrieved facts float to the top
    print("\nTop-5 by importance (hit_count × base_importance):")
    ranked = sorted(lt3._records, key=lambda r: r.importance * r.hit_count, reverse=True)[:5]
    for rec in ranked:
        print(f"  hits={rec.hit_count:2d}  imp={rec.importance:.2f}  {rec.text[:60]}")

    lt3.save(STORE)
    print(f"\nFinal long-term size: {lt3.size} | store: {STORE}")


if __name__ == "__main__":
    main()
