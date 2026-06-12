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

# ─── Reconhecimento facial (CNN própria, treinada no LFW) ────────────────
MODELS_DIR = DATA_DIR / "models"
EMBEDDING_MODEL_PATH = MODELS_DIR / "face_embedding.pt"

EMBEDDING_DIM = 128                 # tamanho do vetor de embedding gerado pela rede
EMBEDDING_BACKBONE = "mobilenet_v2"  # "mobilenet_v2" (leve) ou "resnet18" (mais pesado)
EMBEDDING_IMAGE_SIZE = 112           # lado (px) do recorte do rosto enviado à rede

# ─── Treino (uv run main.py train) ───────────────────────────────────────
LFW_MIN_FACES_PER_PERSON = 20  # mínimo de fotos/pessoa no LFW (menos = mais
                                # pessoas/classes, melhor generalização, treino mais lento)
TRAIN_EPOCHS = 8
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-3
TRAIN_VAL_SPLIT = 0.15

# Limiar de similaridade de cosseno (0–1). MAIOR = mais rígido (menos falsos positivos).
# Com o ArcFace pronto, 0.35–0.50 funcionava bem. Com a NOSSA rede a distribuição das
# similaridades é mais "achatada" (pessoas diferentes já ficam por volta de 0.28 em
# média). Calibrado empiricamente sobre as fotos de data/dataset/ após o treino padrão
# (8 épocas / 62 pessoas do LFW): 0.75 deu ~87% de acerto entre "mesma pessoa" vs
# "pessoa diferente", priorizando poucos falsos positivos. Se trocar o treino/dataset,
# recalibre observando `FaceRecognizer.identify_per_person`.
RECOGNITION_THRESHOLD = 0.75

# Desempenho: o reconhecimento (nossa CNN, em PyTorch) roda numa thread em segundo
# plano (ver src/app.py), então o vídeo e a prova de vida não travam esperando o
# modelo.

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
