"""
A prova de vida aqui é baseada em **marcos faciais (landmarks)** do MediaPipe.
Calculamos o **EAR (Eye Aspect Ratio)** — a razão entre a altura e a largura do
olho. Quando o olho fecha, a altura cai e o EAR despenca; quando reabre, o EAR
volta a subir. Essa transição (alto → baixo → alto) conta como uma **piscada**.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

Uma foto estática não pisca, então exigir 1 piscada já barra o ataque mais
comum de spoofing por foto.

Implementação: usamos a API **MediaPipe Tasks** (`FaceLandmarker`), a mesma
família de API usada na captura (`FaceDetector`).
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from . import config

# Modelo de marcos faciais (baixado automaticamente na primeira execução).
FACE_LANDMARKER_MODEL = config.SRC_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Índices dos 6 pontos de cada olho na malha facial (468/478 landmarks).
# Ordem: [canto_externo, topo1, topo2, canto_interno, base2, base1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def _ensure_model() -> None:
    """Baixa o modelo FaceLandmarker do MediaPipe caso ainda não exista."""
    if not FACE_LANDMARKER_MODEL.exists():
        FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"📥 Baixando modelo FaceLandmarker em: {FACE_LANDMARKER_MODEL}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("✅ Modelo de marcos faciais baixado com sucesso!")


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_aspect_ratio(landmarks, indices, w: int, h: int) -> float:
    """Calcula o EAR de um olho a partir de 6 landmarks (em pixels)."""
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    p1, p2, p3, p4, p5, p6 = p
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class LivenessDetector:
    """
    Detector de piscadas com máquina de estados simples.

    Uso:
        live = LivenessDetector()
        ear, blinks, face_found = live.update(frame_bgr)
        if blinks >= config.REQUIRED_BLINKS: ...  # passou na prova de vida
        live.reset()  # zera a contagem ao iniciar uma nova tentativa
    """

    def __init__(
        self,
        ear_threshold: float | None = None,
        consec_frames: int | None = None,
    ):
        self.ear_threshold = (
            ear_threshold if ear_threshold is not None else config.EAR_THRESHOLD
        )
        self.consec_frames = (
            consec_frames if consec_frames is not None else config.EAR_CONSEC_FRAMES
        )

        _ensure_model()

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(FACE_LANDMARKER_MODEL)
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=2,                      # detecta até 2 rostos (segurança: recusa múltiplos)
            min_face_detection_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self.reset()

    # ─── Estado ──────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Zera contagem de piscadas e o estado interno (frames fechados)."""
        self.blinks = 0
        self._closed_frames = 0
        self.last_ear = 0.0

    # ─── Atualização por frame ────────────────────────────────────────────────────
    def update(self, frame_bgr):
        """
        Processa um frame e atualiza a contagem de piscadas.

        Retorna (ear, blinks, face_count, bbox):
          - ear: razão atual (None se nenhum rosto)
          - blinks: total de piscadas acumuladas desde o último reset()
          - face_count: nº de rostos encontrados neste frame (0, 1 ou 2)
          - bbox: (x1, y1, x2, y2) do 1º rosto em pixels, ou None
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        if face_count == 0:
            self._closed_frames = 0
            return None, self.blinks, 0, None

        landmarks = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]

        # Bounding box a partir dos marcos faciais (min/max das coordenadas).
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        bbox = (
            int(min(xs) * w), int(min(ys) * h),
            int(max(xs) * w), int(max(ys) * h),
        )

        ear = (
            _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            + _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        ) / 2.0
        self.last_ear = ear

        # Máquina de estados: conta frames com olho fechado; ao reabrir após
        # alguns frames fechados, registra uma piscada.
        if ear < self.ear_threshold:
            self._closed_frames += 1
        else:
            if self._closed_frames >= self.consec_frames:
                self.blinks += 1
            self._closed_frames = 0

        return ear, self.blinks, face_count, bbox

    def close(self) -> None:
        """Libera o recurso do MediaPipe."""
        try:
            self._landmarker.close()
        except Exception:
            pass
