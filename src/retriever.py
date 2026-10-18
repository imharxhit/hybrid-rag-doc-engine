import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_\-]+", text.lower())

def chunk_document(doc_id: str, page: int, text: str, chunk_size: int = 512, overlap: int = 64) -> List[DocumentChunk]:
    words = text.split()
    step = max(1, chunk_size - overlap)
    return [
        DocumentChunk(f"{doc_id}_p{page}_c{i}", doc_id, page, " ".join(words[s:s + chunk_size]))
        for i, s in enumerate(range(0, len(words), step)) if words[s:s + chunk_size]
    ]

class HybridRRFEngine:
    def __init__(self, rrf_k: int = 60, k1: float = 1.5, b: float = 0.75):
        self.rrf_k, self.k1, self.b = rrf_k, k1, b
        self.chunks: List[DocumentChunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Counter = Counter()
        self.avg_dl: float = 0.0
        self.embeddings: np.ndarray = np.empty((0, 0))

    def _embed(self, text: str, dim: int = 128) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in tokenize(text):
            vec[hash(tok) % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        self.chunks.extend(chunks)
        self.doc_tokens = [tokenize(c.text) for c in self.chunks]
        self.doc_freqs = Counter(tok for doc in self.doc_tokens for tok in set(doc))
        self.avg_dl = sum(len(d) for d in self.doc_tokens) / max(1, len(self.chunks))
        self.embeddings = np.vstack([self._embed(c.text) for c in self.chunks])
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        q_tokens = tokenize(query)
        n_docs = len(self.chunks)
        bm25 = [0.0] * n_docs
        for q in q_tokens:
            df = self.doc_freqs.get(q, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i, d_toks in enumerate(self.doc_tokens):
                tf = d_toks.count(q)
                if tf:
                    bm25[i] += idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (len(d_toks) / self.avg_dl))))
        dense = (self.embeddings @ self._embed(query)).tolist()
        bm25_rank = sorted(range(n_docs), key=lambda i: bm25[i], reverse=True)
        dense_rank = sorted(range(n_docs), key=lambda i: dense[i], reverse=True)
        rrf: Dict[int, float] = {i: 0.0 for i in range(n_docs)}
        for r, idx in enumerate(bm25_rank, 1):
            rrf[idx] += 1.0 / (self.rrf_k + r)
        for r, idx in enumerate(dense_rank, 1):
            rrf[idx] += 1.0 / (self.rrf_k + r)
        best = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
        return [(self.chunks[i], round(rrf[i], 5)) for i in best]
