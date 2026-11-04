"""
RAG knowledge base for maintenance guidance.

Chunks live as Markdown files with a small front-matter header in
`src/copilot/knowledge/` (one chunk per file):

    ---
    id: "crack-high"
    defect_type: "crack"          # crack | corrosion | dent | rivet_damage | general
    severity: "HIGH"              # HIGH | MEDIUM | LOW | ANY
    source: "..."                 # where the guidance comes from
    action: "..."                 # one-line recommended action
    interval: "within 24 hours"   # optional
    ---
    body text ...

Retrieval is *hybrid*: embedding similarity (sentence-transformers + FAISS
when installed, otherwise TF-IDF cosine so the app still runs offline) plus
a metadata boost for matching defect type / severity. Every retrieved chunk
keeps its `id` so reports can cite it and the verifier can check the citation.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from . import config

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
REQUIRED_CHUNK_IDS = {"no-defects", "engine-health", "progression-growth", "advisory-limits"}


class MissingKnowledgeChunksError(RuntimeError):
    """Raised when required guidance files are missing from the knowledge base."""


def check_required_chunks(chunks: List["Chunk"], directory: str | None = None) -> None:
    present = {chunk.id for chunk in chunks}
    missing = sorted(REQUIRED_CHUNK_IDS - present)
    if missing:
        folder = os.path.abspath(directory or config.KNOWLEDGE_DIR)
        raise MissingKnowledgeChunksError(
            f"Missing required knowledge chunk(s): {', '.join(missing)}. "
            f"Knowledge folder: {folder}"
        )


@dataclass
class Chunk:
    id: str
    title: str
    defect_type: str
    severity: str
    source: str
    action: str
    interval: str
    text: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def index_text(self) -> str:
        # What gets embedded: metadata words help short queries match.
        return f"{self.title}. {self.defect_type} {self.severity}. {self.text} Action: {self.action}"


@dataclass
class Retrieved:
    chunk: Chunk
    score: float

    def to_dict(self) -> Dict:
        d = self.chunk.to_dict()
        d["score"] = round(float(self.score), 3)
        return d


def _parse_chunk_file(path: str) -> Optional[Chunk]:
    raw = open(path, encoding="utf-8").read()
    m = _FM_RE.match(raw)
    if not m:
        return None
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"').strip("'")
    if "id" not in meta:
        return None
    text = re.sub(r"\s+", " ", m.group(2)).strip()
    return Chunk(
        id=meta["id"], title=meta.get("title", meta["id"]),
        defect_type=meta.get("defect_type", "general"),
        severity=meta.get("severity", "ANY").upper(),
        source=meta.get("source", "unknown"),
        action=meta.get("action", ""), interval=meta.get("interval", ""),
        text=text,
    )


def load_chunks(directory: str | None = None) -> List[Chunk]:
    directory = directory or config.KNOWLEDGE_DIR
    chunks = []
    for p in sorted(glob.glob(os.path.join(directory, "*.md"))):
        c = _parse_chunk_file(p)
        if c:
            chunks.append(c)
    return chunks


class KnowledgeBase:
    """Hybrid retriever over the maintenance-guidance chunks."""

    TYPE_BOOST = 0.35
    SEVERITY_BOOST = 0.25
    ANY_SEVERITY_BOOST = 0.08

    def __init__(self, directory: str | None = None, backend: str = "auto"):
        self.chunks = load_chunks(directory)
        check_required_chunks(self.chunks, directory)
        if not self.chunks:
            raise RuntimeError(f"No knowledge chunks found in {directory or config.KNOWLEDGE_DIR}")
        self.by_id = {c.id: c for c in self.chunks}
        self.backend = "tfidf"
        self._vec = None
        self._matrix = None
        self._faiss = None
        self._st = None
        self._build(backend)

    # ------------------------------------------------------------ index
    def _build(self, backend: str) -> None:
        texts = [c.index_text for c in self.chunks]
        if backend in ("auto", "embeddings"):
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(config.EMBED_MODEL)
                emb = self._st.encode(texts, normalize_embeddings=True).astype("float32")
                try:
                    import faiss
                    index = faiss.IndexFlatIP(emb.shape[1])
                    index.add(emb)
                    self._faiss = index
                    self.backend = "sentence-transformers+faiss"
                except ImportError:
                    self._matrix = emb
                    self.backend = "sentence-transformers+numpy"
                return
            except Exception:
                if backend == "embeddings":
                    raise
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vec.fit_transform(texts)
        self.backend = "tfidf"

    def _similarities(self, query: str) -> np.ndarray:
        if self._st is not None:
            q = self._st.encode([query], normalize_embeddings=True).astype("float32")
            if self._faiss is not None:
                scores, idx = self._faiss.search(q, len(self.chunks))
                out = np.zeros(len(self.chunks), dtype=np.float32)
                out[idx[0]] = scores[0]
                return out
            return (self._matrix @ q[0]).astype(np.float32)
        q = self._vec.transform([query])
        return np.asarray((self._matrix @ q.T).todense()).ravel().astype(np.float32)

    # ----------------------------------------------------------- search
    def search(self, query: str, k: int = 4, defect_type: str | None = None,
               severity: str | None = None) -> List[Retrieved]:
        sims = self._similarities(query)
        sev = severity.upper() if severity else None
        scored = []
        for c, s in zip(self.chunks, sims):
            score = float(s)
            if defect_type and c.defect_type == defect_type:
                score += self.TYPE_BOOST
            if sev:
                if c.severity == sev:
                    score += self.SEVERITY_BOOST
                elif c.severity == "ANY":
                    score += self.ANY_SEVERITY_BOOST
            scored.append(Retrieved(c, score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def for_defect(self, defect_type: str, severity: str, trend: str | None = None,
                   k: int = 3) -> List[Retrieved]:
        """Guidance for one finding: severity-specific + general chunk for its type
        (+ the progression chunk when it is growing/new)."""
        query = f"{defect_type.replace('_', ' ')} {severity} severity recommended maintenance action inspection interval"
        hits = self.search(query, k=k, defect_type=defect_type, severity=severity)
        if trend in ("growing", "new") and not any(h.chunk.id == "progression-growth" for h in hits):
            if "progression-growth" in self.by_id:
                hits.append(Retrieved(self.by_id["progression-growth"], 0.0))
        return hits

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.by_id.get(chunk_id)


_default_kb: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb
