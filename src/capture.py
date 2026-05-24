from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from . import config


# ─── Modelo ──────────────────────────────────────────────────────────────────────
def ensure_model() -> None:
    model_path = Path(config.MP_DETECTOR_MODEL)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"📥 Baixando modelo MediaPipe em: {model_path}")
        urllib.request.urlretrieve(config.MP_DETECTOR_MODEL_URL, model_path)
        print("✅ Modelo baixado com sucesso!")


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def extract_bbox(detection, frame_shape, margin: float = 0.0):
    """
    Retorna (x_min, y_min, x_max, y_max) em pixels, já clampado ao frame.

    `margin` adiciona uma folga proporcional ao redor do rosto — útil para o
    recorte não ficar "colado" no rosto, o que melhora a detecção/alinhamento
    na etapa de embeddings.
    """
    h, w = frame_shape[:2]
    bbox = detection.bounding_box

    bw, bh = bbox.width, bbox.height
    mx, my = int(bw * margin), int(bh * margin)

    x_min = max(0, int(bbox.origin_x) - mx)
    y_min = max(0, int(bbox.origin_y) - my)
    x_max = min(w, int(bbox.origin_x + bw) + mx)
    y_max = min(h, int(bbox.origin_y + bh) + my)
    return x_min, y_min, x_max, y_max


def draw_detections(frame, detections) -> None:
    h, w = frame.shape[:2]
    for detection in detections:
        x_min, y_min, x_max, y_max = extract_bbox(detection, frame.shape)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if detection.keypoints:
            for kp in detection.keypoints:
                kp_x = int(kp.x * w)   # keypoints são normalizados (0–1)
                kp_y = int(kp.y * h)
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)


# ─── Loop principal de captura ────────────────────────────────────────────────────
def run_capture(person_name: str | None = None) -> None:
    person = person_name or config.PERSON_NAME
    dataset_dir = Path(config.DATASET_DIR) / person
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ensure_model()

    print(f"📁 Fotos serão salvas em: {dataset_dir}")
    print("🎥 Iniciando captura...")
    print("Controles:")
    print("  📸 'C' para capturar e salvar a foto")
    print("  ❌ 'Q' para sair")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Erro: não foi possível acessar a webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    photo_count = len(list(dataset_dir.glob("*.jpg")))

    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.MP_DETECTOR_MODEL)),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    )

    try:
        with FaceDetector.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao ler frame da câmera.")
                    break

                if config.MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = detector.detect(mp_image)

                display = frame.copy()
                if results.detections:
                    draw_detections(display, results.detections)

                cv2.putText(display, f"Fotos salvas: {photo_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "'C' capturar | 'Q' sair", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("FaceID - Captura e Detecção", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    print("Saindo...")
                    break

                elif key in (ord("c"), ord("C")):
                    if results.detections:
                        x_min, y_min, x_max, y_max = extract_bbox(
                            results.detections[0], frame.shape, margin=config.CAPTURE_MARGIN
                        )
                        face_crop = frame[y_min:y_max, x_min:x_max]

                        if face_crop.size == 0:
                            print("⚠️  Recorte do rosto inválido. Tente novamente.")
                        else:
                            photo_count += 1
                            filename = dataset_dir / f"{photo_count:03d}.jpg"
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
