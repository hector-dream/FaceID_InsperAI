"""
Configuração central do FaceID — Insper AI 2026.1
==================================================

Todos os parâmetros ajustáveis do projeto ficam aqui para facilitar o
"fine-tuning". Os outros módulos importam deste
arquivo, então mudar um valor aqui reflete em todo o pipeline.
"""

from pathlib import Path

# ─── Diretórios ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"                  # fotos cadastradas
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"       # base de vetores

PERSON_NAME = "anderson"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_PREVIEW = True 

# ─── Detecção facial (MediaPipe FaceDetector / BlazeFace) ────────────────
MP_DETECTOR_MODEL = SRC_DIR / "blaze_face_short_range.tflite"
MP_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MIN_DETECTION_CONFIDENCE = 0.5
CAPTURE_MARGIN = 0.40          # margem (proporcional) ao redor do rosto no recorte
                               # (margem generosa ajuda o InsightFace a detectar no cadastro)

# ─── Reconhecimento facial (InsightFace / ArcFace) ─────────────────────
# "buffalo_l" = melhor precisão | "buffalo_s" = mais leve/rápido em CPUs modestas.
INSIGHTFACE_MODEL_PACK = "buffalo_l"
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]  
INSIGHTFACE_CTX_ID = 0
INSIGHTFACE_DET_SIZE = (320, 320)   # menor = detecção mais rápida em CPU (não afeta o cadastro já feito)

# Limiar de similaridade de cosseno (0–1). MAIOR = mais rígido (menos falsos positivos).
# ArcFace costuma separar bem por volta de 0.35–0.50.
RECOGNITION_THRESHOLD = 0.40

# Desempenho: o reconhecimento (InsightFace, pesado) roda numa thread em segundo
# plano (ver src/app.py), então o vídeo e a prova de vida não travam esperando o
# modelo. Para acelerar ainda mais o reconhecimento, troque o pacote para
# "buffalo_s" acima — mas aí é preciso refazer o cadastro (uv run main.py enroll).

# ─── Prova de vida / Anti-spoofing (piscada via MediaPipe FaceMesh) ──────
EAR_THRESHOLD = 0.21          # abaixo disso o olho é considerado fechado (EAR)
EAR_CONSEC_FRAMES = 1         # frames consecutivos fechados para validar 1 piscada
REQUIRED_BLINKS = 1           # piscadas necessárias para liberar o acesso
LIVENESS_TIMEOUT_S = 7.0      # tempo para piscar após ser reconhecido (reinicia depois)

# ─── Qualidade, exceções e interface ─────────────────────────────────────
MIN_FACE_AREA_RATIO = 0.030   # área mínima do rosto / frame  => "aproxime-se"
MIN_BRIGHTNESS = 55           # brilho médio mínimo do frame (0–255) => "iluminação ruim"
GRANTED_DISPLAY_S = 3.0       # tempo que a tela verde de "acesso permitido" permanece
WINDOW_NAME = "FaceID - Insper AI"
START_FULLSCREEN = False
