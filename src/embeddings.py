"""
Dia 2 — Extração de características (embeddings)
===============================================

"Apenas detectar o rosto não basta; precisamos transformá-lo em números que o
computador entenda. Rostos parecidos geram números parecidos."

Aqui usamos o InsightFace (modelo ArcFace, pacote `buffalo_l`) que recebe a
imagem de um rosto e devolve um vetor de 512 dimensões (o *embedding*). O
InsightFace também faz a detecção e o alinhamento do rosto internamente, o que
melhora bastante a qualidade do vetor em comparação a recortar "na mão".

Este módulo:
  1. Carrega o modelo InsightFace uma única vez (cache global).
  2. Gera o embedding de uma imagem qualquer.
  3. Varre `data/dataset/<pessoa>/*.jpg` (fotos do Dia 1) e monta a base de
     conhecidos, salvando em `data/embeddings.pkl`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from . import config

# ─── Carregamento do modelo (lazy + cache) ──────────────────────────────────────
_APP = None


def get_face_app():
    """
    Carrega o InsightFace apenas uma vez e reutiliza nas próximas chamadas.

    Importamos o `insightface` dentro da função para que módulos que não
    precisam de reconhecimento (ex.: só captura do Dia 1) não falhem caso a
    biblioteca ainda não esteja instalada.
    """
    global _APP
    if _APP is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "InsightFace não está instalado. Rode:\n"
                "    uv pip install insightface onnxruntime\n"
                "(no Windows pode ser necessário o 'Microsoft C++ Build Tools')."
            ) from exc

        print(f"🧠 Carregando modelo InsightFace '{config.INSIGHTFACE_MODEL_PACK}'...")
        app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_PACK,
            providers=config.INSIGHTFACE_PROVIDERS,
        )
        app.prepare(
            ctx_id=config.INSIGHTFACE_CTX_ID,
            det_size=config.INSIGHTFACE_DET_SIZE,
        )
        _APP = app
        print("✅ Modelo de reconhecimento pronto.")
    return _APP


# ─── Extração de embeddings ─────────────────────────────────────────────────────
def detect_faces(image_bgr):
    """Roda o InsightFace e retorna a lista de rostos (objetos Face)."""
    app = get_face_app()
    return app.get(image_bgr)


def largest_face(faces):
    """Retorna o rosto de maior área de uma lista (ou None se vazia)."""
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def detect_faces_robust(image_bgr):
    """
    Detecção tolerante a recortes "justos" (usada no cadastro).

    Recortes do Dia 1 muitas vezes têm o rosto preenchendo a imagem inteira,
    sem contexto ao redor — e o detector do InsightFace (SCRFD) costuma falhar
    nesse caso. Aqui tentamos, em ordem:
      1. a imagem como veio;
      2. ampliada (se for pequena);
      3. com uma borda ao redor, dando "contexto" para o detector.
    """
    app = get_face_app()

    faces = app.get(image_bgr)
    if faces:
        return faces

    img = image_bgr
    h, w = img.shape[:2]

    # 2) Amplia imagens pequenas (o detector trabalha melhor com mais pixels).
    if min(h, w) < 200:
        scale = 200.0 / min(h, w)
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
        faces = app.get(img)
        if faces:
            return faces

    # 3) Adiciona uma borda (replicando as bordas) para dar contexto ao rosto.
    pad = int(max(img.shape[:2]) * 0.6)
    padded = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    return app.get(padded)


def embedding_from_image(image_bgr):
    """
    Retorna o embedding (vetor 512-d normalizado) do maior rosto da imagem.
    Retorna None se nenhum rosto for encontrado.
    """
    face = largest_face(detect_faces_robust(image_bgr))
    if face is None:
        return None
    return np.asarray(face.normed_embedding, dtype=np.float32)


# ─── Construção da base de conhecidos ────────────────────────────────────────────
def build_database(dataset_dir=None, output_path=None, verbose=True):
    """
    Percorre o dataset (uma subpasta por pessoa), gera os embeddings de cada foto
    e salva a base em um arquivo .pkl.

    Estrutura salva (fácil de consumir no Dia 3):
        {
            "names":      ["anderson", "anderson", "lara", ...],   # 1 por embedding
            "embeddings": np.ndarray de shape (N, 512), já normalizados,
        }
    """
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    output_path = Path(output_path or config.EMBEDDINGS_PATH)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {dataset_dir}.\n"
            "Rode primeiro o cadastro do Dia 1:  python main.py capture"
        )

    person_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not person_dirs:
        raise FileNotFoundError(
            f"Nenhuma pessoa cadastrada em {dataset_dir}. "
            "Cadastre rostos com:  python main.py capture"
        )

    names: list[str] = []
    embeddings: list[np.ndarray] = []

    for person_dir in person_dirs:
        person = person_dir.name
        images = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        saved = 0

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                if verbose:
                    print(f"⚠️  Não consegui abrir {img_path.name}, pulando.")
                continue

            emb = embedding_from_image(image)
            if emb is None:
                if verbose:
                    print(f"⚠️  Nenhum rosto detectado em {img_path.name}, pulando.")
                continue

            names.append(person)
            embeddings.append(emb)
            saved += 1

        if verbose:
            print(f"👤 {person}: {saved} embedding(s) gerado(s) de {len(images)} foto(s).")

    if not embeddings:
        raise RuntimeError(
            "Nenhum embedding foi gerado. Verifique se as fotos do dataset "
            "contêm rostos visíveis e bem iluminados."
        )

    database = {
        "names": names,
        "embeddings": np.vstack(embeddings).astype(np.float32),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(database, f)

    if verbose:
        n_people = len(set(names))
        print(
            f"\n💾 Base salva em {output_path}\n"
            f"   {len(names)} embeddings | {n_people} pessoa(s)."
        )
    return database


if __name__ == "__main__":
    build_database()
