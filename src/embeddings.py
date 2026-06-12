"""
Este módulo:
  1. Carrega o detector de rostos (MediaPipe) e a nossa rede de embeddings
     (PyTorch, treinada com `uv run main.py train`) uma única vez (cache global).
  2. Gera o embedding de uma imagem qualquer.
  3. Varre `data/dataset/<pessoa>/*.jpg` e monta a base de
     conhecidos, salvando em `data/embeddings.pkl`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from . import config

# ─── Carregamento do modelo (lazy + cache) ──────────────────────────────────────
_APP = None


class _Face:
    """
    Objeto leve que imita a interface usada pelo resto do projeto
    (`face.bbox`, `face.normed_embedding`) — o mesmo formato que o
    InsightFace expunha antes.
    """

    __slots__ = ("bbox", "normed_embedding")

    def __init__(self, bbox: np.ndarray, normed_embedding: np.ndarray):
        self.bbox = bbox
        self.normed_embedding = normed_embedding


class _FaceApp:
    """Detector de rostos (MediaPipe FaceDetector) + rede de embeddings (nossa CNN)."""

    def __init__(self):
        import torch
        from torchvision import transforms

        from .capture import ensure_model, extract_bbox
        from .face_model import FaceEmbeddingNet

        if not config.EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo de embeddings não encontrado em {config.EMBEDDING_MODEL_PATH}.\n"
                "Treine-o primeiro:  uv run main.py train"
            )

        ensure_model()  # garante o modelo de detecção do MediaPipe (mesmo do capture.py)
        self._extract_bbox = extract_bbox

        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(config.MP_DETECTOR_MODEL)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        )
        self._detector = FaceDetector.create_from_options(options)

        checkpoint = torch.load(config.EMBEDDING_MODEL_PATH, map_location="cpu")
        net = FaceEmbeddingNet(
            embedding_dim=checkpoint["embedding_dim"],
            backbone=checkpoint["backbone"],
            pretrained=False,
        )
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        self._torch = torch
        self._net = net

        image_size = checkpoint["image_size"]
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get(self, image_bgr):
        """Detecta rostos e retorna uma lista de `_Face` (bbox + embedding normalizado)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        faces: list[_Face] = []
        for detection in result.detections or []:
            x1, y1, x2, y2 = self._extract_bbox(detection, image_bgr.shape, margin=config.CAPTURE_MARGIN)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            embedding = self._embed(crop)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            faces.append(_Face(bbox=bbox, normed_embedding=embedding))
        return faces

    def _embed(self, face_bgr) -> np.ndarray:
        """Roda a CNN no recorte do rosto e devolve o embedding L2-normalizado."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0)
        with self._torch.no_grad():
            embedding = self._net(tensor)
        embedding = embedding.squeeze(0).numpy().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


def get_face_app() -> _FaceApp:
    """Carrega o detector + a CNN de embeddings apenas uma vez e reutiliza nas próximas chamadas."""
    global _APP
    if _APP is None:  # lazy load - primeira chamada
        print("🧠 Carregando modelo de reconhecimento facial (CNN própria)...")
        _APP = _FaceApp()
        print("✅ Modelo de reconhecimento pronto.")
    return _APP


# ─── Extração de embeddings ─────────────────────────────────────────────────────
def detect_faces(image_bgr):
    """Roda a detecção + embeddings e retorna a lista de rostos (`_Face`)."""
    app = get_face_app()
    return app.get(image_bgr)


def largest_face(faces):
    """Retorna o rosto de maior área de uma lista (ou None se vazia)."""
    if not faces:
        return None
    # em geral o maior = o mais perto; mais confiável para embeddings
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def detect_faces_robust(image_bgr):
    """
    Detecção tolerante a recortes "justos" (usada no cadastro).

    Recortes do Dia 1 muitas vezes têm o rosto preenchendo a imagem inteira,
    sem contexto ao redor — e o detector (MediaPipe) costuma falhar nesse
    caso. Aqui tentamos, em ordem:
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
    Retorna o embedding (vetor normalizado, dimensão `config.EMBEDDING_DIM`)
    do maior rosto da imagem. Retorna None se nenhum rosto for encontrado.
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

    Estrutura salva:
        {
            "names":      ["anderson", "anderson", "lara", ...],
            "embeddings": np.ndarray de shape (N, config.EMBEDDING_DIM), já normalizados,
        }
    """
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    output_path = Path(output_path or config.EMBEDDINGS_PATH)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {dataset_dir}.\n"
            "Rode primeiro o cadastro do:  uv run main.py capture"
        )

    person_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not person_dirs:
        raise FileNotFoundError(
            f"Nenhuma pessoa cadastrada em {dataset_dir}. "
            "Cadastre rostos com:  uv run main.py capture"
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
