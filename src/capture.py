import cv2
import mediapipe as mp
import urllib.request
from pathlib import Path

# ─── Configurações ───────────────────────────────────────────────────────────

PERSON_NAME = "hector"  # Mude para o nome da pessoa
DATASET_DIR = Path(__file__).parent.parent / "data" / "dataset" / PERSON_NAME
MODEL_PATH = Path(__file__).parent / "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

def ensure_dataset_dir():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"📥 Baixando modelo MediaPipe em: {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modelo baixado com sucesso!")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_bbox(detection, frame_shape):
    """
    Retorna (x_min, y_min, x_max, y_max) em pixels, já clampado.

    No MediaPipe Tasks (FaceDetector), as coordenadas da BoundingBox
    são fornecidas diretamente em pixels — NÃO em proporção (0–1).
    Por isso não multiplicamos por largura/altura do frame.
    """
    h, w = frame_shape[:2]
    bbox = detection.bounding_box

    x_min = max(0, int(bbox.origin_x))
    y_min = max(0, int(bbox.origin_y))
    x_max = min(w, int(bbox.origin_x + bbox.width))
    y_max = min(h, int(bbox.origin_y + bbox.height))

    return x_min, y_min, x_max, y_max


def draw_detections(frame, detections):
    """Desenha bounding boxes e keypoints no frame."""
    h, w = frame.shape[:2]

    for detection in detections:
        x_min, y_min, x_max, y_max = extract_bbox(detection, frame.shape)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if detection.keypoints:
            for kp in detection.keypoints:
                # Keypoints também são em pixels no MediaPipe Tasks
                kp_x = int(kp.x * w)  # keypoints SIM são normalizados (0–1)
                kp_y = int(kp.y * h)
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)

def run_capture():
    ensure_dataset_dir()
    ensure_model()

    print(f"📁 Fotos serão salvas em: {DATASET_DIR}")
    print("🎥 Iniciando captura...")
    print("Controles:")
    print("  📸 Pressione 'C' para capturar e salvar a foto")
    print("  ❌ Pressione 'Q' para sair")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Erro: não foi possível acessar a webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    photo_count = len(list(DATASET_DIR.glob("*.jpg")))

    # Configura o detector
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    try:
        with FaceDetector.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao ler frame da câmera.")
                    break

                # Detecção
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = detector.detect(mp_image)

                # Visualização
                display = frame.copy()
                if results.detections:
                    draw_detections(display, results.detections)

                cv2.putText(display, f"Fotos salvas: {photo_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Pressione 'C' para capturar | 'Q' para sair", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("FaceID - Captura e Detecção", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Saindo...")
                    break

                elif key in (ord("c"), ord("C")):
                    if results.detections:
                        x_min, y_min, x_max, y_max = extract_bbox(
                            results.detections[0], frame.shape
                        )
                        face_crop = frame[y_min:y_max, x_min:x_max]

                        if face_crop.size == 0:
                            print("⚠️  Recorte do rosto inválido. Tente novamente.")
                        else:
                            photo_count += 1
                            filename = DATASET_DIR / f"{photo_count:03d}.jpg"
                            cv2.imwrite(str(filename), face_crop)
                            print(f"✅ Foto #{photo_count} salva: {filename}")
                    else:
                        print("⚠️  Nenhum rosto detectado. Tente novamente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Captura finalizada!")


if __name__ == "__main__":
    run_capture()