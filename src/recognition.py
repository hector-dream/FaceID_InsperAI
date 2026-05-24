"""
Quando um novo rosto aparece, o InsightFace gera um embedding temporário.
Comparamos esse vetor com os vetores salvos na base usando a
**similaridade de cosseno**:

    sim(a, b) = (a · b) / (||a|| · ||b||)

Como os embeddings do InsightFace já vêm normalizados (norma 1), a similaridade
de cosseno vira simplesmente o produto escalar `a · b`, variando de -1 a 1
(quanto MAIOR, mais parecidos os rostos).

> Obs.: a rubrica cita distância euclidiana com limiar ~0.6 (mundo dlib). Com
> ArcFace/cosseno a lógica é equivalente, só que o limiar é de SIMILARIDADE
> (maior = mais parecido) — por isso comparamos com `>=` em vez de `<=`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import config


class FaceRecognizer:
    """Carrega a base de embeddings e identifica novos rostos por cosseno."""

    def __init__(self, db_path=None, threshold: float | None = None):
        self.db_path = Path(db_path or config.EMBEDDINGS_PATH)
        self.threshold = (
            threshold if threshold is not None else config.RECOGNITION_THRESHOLD
        )
        self.names: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.load()

    # ─── Carregamento da base ───────────────────────────────────────────────────
    def load(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Base de embeddings não encontrada em {self.db_path}.\n"
                "Gere-a primeiro:  uv run main.py enroll"
            )
        with open(self.db_path, "rb") as f:
            db = pickle.load(f)

        self.names = list(db["names"])
        self.embeddings = np.asarray(db["embeddings"], dtype=np.float32)

        # Garante que toda a base está normalizada (segurança extra).
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    @property
    def people(self) -> list[str]:
        """Lista de pessoas distintas cadastradas."""
        return sorted(set(self.names))

    # ─── Identificação ──────────────────────────────────────────────────────────
    def identify(self, embedding) -> tuple[str, float]:
        """
        Recebe um embedding (512-d) e retorna `(nome, score)`.

        `score` é a maior similaridade de cosseno encontrada. Se ela ficar abaixo
        do `threshold`, devolvemos "Desconhecido".
        """
        if embedding is None or self.embeddings.shape[0] == 0:
            return "Desconhecido", 0.0

        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm == 0:
            return "Desconhecido", 0.0
        query = query / norm

        # Produto escalar de todos os embeddings da base contra a query.
        sims = self.embeddings @ query          # shape (N,)
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score >= self.threshold:
            return self.names[best], score
        return "Desconhecido", score

    def identify_per_person(self, embedding) -> dict[str, float]:
        """
        Retorna a melhor similaridade por pessoa — útil para depurar/ajustar o threshold.
        """
        result: dict[str, float] = {}
        if embedding is None or self.embeddings.shape[0] == 0:
            return result

        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm == 0:
            return result
        query = query / norm
        sims = self.embeddings @ query

        for name, sim in zip(self.names, sims):
            sim = float(sim)
            if name not in result or sim > result[name]:
                result[name] = sim
        return result


# ─── Funções utilitárias (independentes da classe) ──────────────────────────────
def cosine_similarity(a, b) -> float:
    """Similaridade de cosseno entre dois vetores (1 = idênticos)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def euclidean_distance(a, b) -> float:
    """Distância euclidiana entre dois vetores (0 = idênticos)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.linalg.norm(a - b))
