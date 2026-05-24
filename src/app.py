"""
Fluxo de autenticação
==================================================

Junta todas as peças num produto visual que simula um Face ID real:

    BLOQUEADO ──(rosto reconhecido)──► PROVA DE VIDA ──(piscou)──► LIBERADO
        ▲                                   │                          │
        └──────(desconhecido / timeout / sem rosto)───────────────────┘

Desempenho (essencial em CPU):
  - A PROVA DE VIDA (MediaPipe FaceLandmarker, leve) roda em TODOS os quadros, no
    loop principal, para a piscada ser captada com fluidez. Ela também informa
    presença de rosto, nº de rostos e a bounding box.
  - O RECONHECIMENTO (InsightFace, pesado) roda numa THREAD em segundo plano,
    sobre o último quadro disponível, e atualiza a identidade em cache. Assim o
    loop principal nunca trava esperando o modelo pesado.

Também trata as exceções pedidas na rubrica:
  - nenhum rosto / múltiplos rostos na câmera
  - rosto muito distante  -> "Aproxime-se da câmera"
  - iluminação ruim       -> "Iluminação insuficiente"
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import config, interface
from .embeddings import detect_faces, get_face_app, largest_face
from .liveness import LivenessDetector
from .recognition import FaceRecognizer


# Estados possíveis
LOCKED = "LOCKED"
LIVENESS = "LIVENESS"
GRANTED = "GRANTED"
DENIED = "DENIED"


# ─── Reconhecimento assíncrono (thread em segundo plano) ─────────────────────────
class RecognitionWorker:
    """
    Processa o reconhecimento (InsightFace) numa thread separada para não travar
    o vídeo. O loop principal envia o quadro mais recente com `submit()` e lê a
    última identidade calculada com `get()`.
    """

    def __init__(self, recognizer: FaceRecognizer):
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._frame = None
        self._result = ("Desconhecido", 0.0)
        self._ready = False              
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()  

    def get(self) -> tuple[str, float]:
        with self._lock:
            return self._result

    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.01)
                continue

            face = largest_face(detect_faces(frame))
            if face is not None:
                res = self.recognizer.identify(face.normed_embedding)
            else:
                res = ("Desconhecido", 0.0)

            with self._lock:
                self._result = res
                self._ready = True

    def stop(self) -> None:
        self._running = False


def _brightness(frame) -> float:
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def _bbox_area_ratio(bbox, frame_shape) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    h, w = frame_shape[:2]
    return area / float(w * h)


def run_auth(threshold: float | None = None) -> None:
    # Recursos de IA
    recognizer = FaceRecognizer(threshold=threshold)
    print(f"👥 Pessoas cadastradas: {', '.join(recognizer.people) or '(nenhuma)'}")
    get_face_app()                       # pré-carrega o InsightFace (antes da thread)
    liveness = LivenessDetector()        # pré-carrega/baixa o FaceLandmarker
    worker = RecognitionWorker(recognizer)

    # Câmera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        worker.stop()
        raise RuntimeError("Erro: não foi possível acessar a webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = config.START_FULLSCREEN
    _apply_fullscreen(fullscreen)

    print("🔒 Sistema iniciado. Controles:  [Q]/[ESC] sair   [F] tela cheia")

    # Estado da máquina
    state = LOCKED
    pending_name: str | None = None    
    liveness_start = 0.0
    granted_name = ""
    granted_time = 0.0

    # FPS (média móvel)
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Erro ao ler frame da câmera.")
                break

            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            handled = False  # garante que cada frame seja desenhado uma única vez

            # ── Tela verde "acesso permitido"
            if state == GRANTED:
                if now - granted_time < config.GRANTED_DISPLAY_S:
                    interface.render_granted(frame, granted_name)
                    handled = True
                else:
                    state = LOCKED
                    pending_name = None
                    liveness.reset()

             # ── Exceção: iluminação ruim
            if not handled and _brightness(frame) < config.MIN_BRIGHTNESS:
                state = LOCKED
                pending_name = None
                liveness.reset()
                interface.render_warning(frame, "Iluminacao insuficiente - melhore a luz")
                handled = True

            if not handled:
                ear, blinks, face_count, bbox = liveness.update(frame)

                if face_count == 0:
                    state = LOCKED
                    pending_name = None
                    liveness.reset()
                    interface.render_locked(frame, "Nenhum rosto detectado")

                elif face_count > 1:
                    # Segurança: recusa se houver mais de uma pessoa na cena.
                    state = LOCKED
                    pending_name = None
                    liveness.reset()
                    interface.render_warning(frame, "Multiplos rostos detectados")

                elif _bbox_area_ratio(bbox, frame.shape) < config.MIN_FACE_AREA_RATIO:
                    state = LOCKED
                    pending_name = None
                    liveness.reset()
                    interface.draw_face_box(frame, bbox, interface.RED)
                    interface.render_warning(frame, "Aproxime-se da camera")

                else:
                    worker.submit(frame)

                    if not worker.ready():
                        state = LOCKED
                        interface.render_locked(frame, "Verificando...")
                    else:
                        name, score = worker.get()

                        if name == "Desconhecido":
                            state = DENIED
                            pending_name = None
                            liveness.reset()
                            interface.render_denied(frame, bbox, score)

                        else:
                            # Reconhecido -> inicia/continua a prova de vida
                            if pending_name != name:
                                pending_name = name
                                liveness.reset()
                                liveness_start = now
                                blinks = 0   # zera após reset (candidato novo)

                            if blinks >= config.REQUIRED_BLINKS:
                                state = GRANTED
                                granted_name = name
                                granted_time = now
                                pending_name = None
                                interface.render_granted(frame, name)
                            else:
                                if now - liveness_start > config.LIVENESS_TIMEOUT_S:
                                    liveness.reset()
                                    liveness_start = now
                                state = LIVENESS
                                interface.render_liveness(
                                    frame, bbox, name, score,
                                    ear, blinks, config.REQUIRED_BLINKS,
                                    config.EAR_THRESHOLD,
                                )

            # ── Exibição + teclado (ÚNICO ponto por iteração) ───────────────────
            _show(frame, fps)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):      
                break
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                _apply_fullscreen(fullscreen)
            if _window_closed():                      
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        liveness.close()
        print("✅ Sistema encerrado.")


def _apply_fullscreen(fullscreen: bool) -> None:
    cv2.setWindowProperty(
        config.WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )


def _show(frame, fps: float) -> None:
    cv2.putText(
        frame, f"FPS: {fps:4.1f}", (frame.shape[1] - 120, 30),
        interface.FONT, 0.6, interface.WHITE, 2, cv2.LINE_AA,
    )
    cv2.imshow(config.WINDOW_NAME, frame)


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


if __name__ == "__main__":
    run_auth()
