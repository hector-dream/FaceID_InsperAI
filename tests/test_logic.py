"""
Testes de lógica pura (sem webcam / sem modelos pesados)
========================================================

Valida o "miolo" matemático do sistema:
  - similaridade de cosseno e distância euclidiana
  - identificação por threshold (FaceRecognizer.identify)
  - cálculo do EAR (prova de vida)

Como rodar (a partir da raiz do projeto):
    python tests/test_logic.py
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

# Permite importar o pacote `src` ao rodar este arquivo diretamente.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recognition import (  # noqa: E402
    FaceRecognizer,
    cosine_similarity,
    euclidean_distance,
)

_passed = 0
_failed = 0


def check(name, condition):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}")


# ─── 1) Similaridade / distância ─────────────────────────────────────────────────
def test_similarity():
    print("Similaridade e distância:")
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])

    check("cosseno de vetores iguais ≈ 1", abs(cosine_similarity(a, b) - 1.0) < 1e-6)
    check("cosseno de vetores ortogonais ≈ 0", abs(cosine_similarity(a, c)) < 1e-6)
    check("distância de vetores iguais ≈ 0", euclidean_distance(a, b) < 1e-6)
    check("distância de ortogonais > 0", euclidean_distance(a, c) > 0.5)


# ─── 2) Identificação por threshold ──────────────────────────────────────────────
def test_recognizer():
    print("Reconhecimento (identify):")
    # Base sintética: 2 pessoas com direções bem diferentes.
    emb_ander = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_lara = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    db = {
        "names": ["anderson", "lara"],
        "embeddings": np.vstack([emb_ander, emb_lara]).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "embeddings.pkl"
        with open(path, "wb") as f:
            pickle.dump(db, f)

        rec = FaceRecognizer(db_path=path, threshold=0.4)

        # Vetor quase idêntico ao do Anderson (com leve ruído).
        query_ander = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        name, score = rec.identify(query_ander)
        check("reconhece a pessoa correta", name == "anderson")
        check("score alto p/ match", score > 0.9)

        # Vetor numa direção nova -> deve ser desconhecido.
        query_unknown = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        name2, score2 = rec.identify(query_unknown)
        check("rejeita desconhecido (abaixo do threshold)", name2 == "Desconhecido")
        check("score baixo p/ desconhecido", score2 < 0.4)

        check("lista de pessoas correta", rec.people == ["anderson", "lara"])


# ─── 3) EAR (prova de vida) ──────────────────────────────────────────────────────
class _LM:
    """Landmark falso com x, y normalizados (imita o MediaPipe)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_ear():
    print("EAR (prova de vida):")
    try:
        from src.liveness import LEFT_EYE, RIGHT_EYE, _eye_aspect_ratio
    except Exception as exc:  # mediapipe/cv2 ausentes
        print(f"  ⚠️  pulando (dependência ausente: {exc})")
        return

    w = h = 1000
    max_idx = max(RIGHT_EYE + LEFT_EYE)
    # Olho ABERTO: cantos afastados na horizontal, pálpebras afastadas na vertical.
    open_lm = [_LM(0.5, 0.5) for _ in range(max_idx + 1)]
    # right eye [canto_ext, topo1, topo2, canto_int, base2, base1]
    p_open = {
        RIGHT_EYE[0]: (0.30, 0.50), RIGHT_EYE[3]: (0.40, 0.50),   # horizontal
        RIGHT_EYE[1]: (0.33, 0.45), RIGHT_EYE[2]: (0.37, 0.45),   # topo
        RIGHT_EYE[5]: (0.33, 0.55), RIGHT_EYE[4]: (0.37, 0.55),   # base
    }
    for idx, (x, y) in p_open.items():
        open_lm[idx] = _LM(x, y)

    ear_open = _eye_aspect_ratio(open_lm, RIGHT_EYE, w, h)

    # Olho FECHADO: pálpebras coladas (altura ~0).
    closed_lm = list(open_lm)
    for idx in (RIGHT_EYE[1], RIGHT_EYE[2], RIGHT_EYE[4], RIGHT_EYE[5]):
        closed_lm[idx] = _LM(closed_lm[idx].x, 0.50)
    ear_closed = _eye_aspect_ratio(closed_lm, RIGHT_EYE, w, h)

    check("EAR aberto > fechado", ear_open > ear_closed)
    check("EAR fechado próximo de 0", ear_closed < 0.1)
    check("EAR aberto razoável (>0.2)", ear_open > 0.2)


if __name__ == "__main__":
    test_similarity()
    test_recognizer()
    test_ear()
    print(f"\nResultado: {_passed} passou(ram), {_failed} falhou(ram).")
    sys.exit(1 if _failed else 0)
