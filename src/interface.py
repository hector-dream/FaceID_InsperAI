"""
Este módulo concentra apenas o DESENHO (a "cara" do sistema). A lógica de
decisão fica em `app.py`. Tudo é desenhado com primitivas do OpenCV, então não
há dependência extra de GUI.

Paleta (BGR, padrão do OpenCV):
  - Vermelho  -> bloqueado / acesso negado
  - Âmbar     -> verificando prova de vida
  - Verde     -> acesso permitido
"""

from __future__ import annotations

import cv2
import numpy as np

# Cores em BGR
RED = (60, 60, 220)
AMBER = (40, 170, 245)
GREEN = (80, 200, 90)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (160, 160, 160)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─── Blocos de desenho reutilizáveis ─────────────────────────────────────────────
def tint(frame, color, alpha: float = 0.28):
    """Aplica um "véu" colorido translúcido sobre todo o frame."""
    overlay = np.full_like(frame, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def _text_centered(frame, text, y, scale, color, thickness=2):
    w = frame.shape[1]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return th


def banner(frame, title, subtitle, color):
    """Faixa superior + título grande centralizado e subtítulo embaixo."""
    h, w = frame.shape[:2]

    # Faixa superior sólida
    cv2.rectangle(frame, (0, 0), (w, 70), color, -1)
    _text_centered(frame, title, 48, 1.1, WHITE, 3)

    # Subtítulo (rodapé)
    cv2.rectangle(frame, (0, h - 48), (w, h), BLACK, -1)
    if subtitle:
        _text_centered(frame, subtitle, h - 16, 0.62, WHITE, 1)
    return frame


def draw_lock(frame, center, size, locked: bool, color):
    """Desenha um cadeado simples (fechado ou aberto) com primitivas."""
    cx, cy = center
    body_w, body_h = size, int(size * 0.8)
    x1, y1 = cx - body_w // 2, cy
    x2, y2 = cx + body_w // 2, cy + body_h

    # Corpo do cadeado
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 2)

    # Arco (haste)
    radius = body_w // 3
    arc_cy = y1
    if locked:
        cv2.ellipse(frame, (cx, arc_cy), (radius, radius), 0, 180, 360, color, 4)
        cv2.line(frame, (cx - radius, arc_cy), (cx - radius, y1), color, 4)
        cv2.line(frame, (cx + radius, arc_cy), (cx + radius, y1), color, 4)
    else:
        # Aberto: haste deslocada para a lateral
        cv2.ellipse(frame, (cx + radius, arc_cy), (radius, radius), 0, 180, 360, color, 4)
        cv2.line(frame, (cx, arc_cy), (cx, y1), color, 4)

    # Buraco da fechadura
    cv2.circle(frame, (cx, cy + body_h // 2), max(3, size // 12), BLACK, -1)
    return frame


def draw_face_box(frame, bbox, color, label=None, score=None):
    """Retângulo no rosto com etiqueta opcional (nome + score)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        text = label if score is None else f"{label} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 6), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    return frame


def draw_ear_bar(frame, ear, threshold, blinks, required):
    """Barrinha de debug do EAR (prova de vida) no canto inferior esquerdo."""
    h = frame.shape[0]
    x, y = 14, h - 70
    cv2.putText(
        frame,
        f"Piscadas: {blinks}/{required}",
        (x, y),
        FONT, 0.6, WHITE, 2, cv2.LINE_AA,
    )
    if ear is not None:
        bar_w = 160
        filled = int(min(ear / 0.4, 1.0) * bar_w)
        cv2.rectangle(frame, (x, y + 12), (x + bar_w, y + 26), GRAY, 1)
        col = GREEN if ear >= threshold else RED
        cv2.rectangle(frame, (x, y + 12), (x + filled, y + 26), col, -1)
        cv2.putText(frame, f"EAR {ear:.2f}", (x + bar_w + 10, y + 25),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


# ─── Telas completas por estado ──────────────────────────────────────────────────
def render_locked(frame, message="Olhe para a camera"):
    tint(frame, RED, 0.30)
    draw_lock(frame, (frame.shape[1] // 2, frame.shape[0] // 2 - 40), 70, True, WHITE)
    banner(frame, "SISTEMA BLOQUEADO", message, RED)
    return frame


def render_denied(frame, bbox=None, score=None, message="Rosto nao cadastrado"):
    tint(frame, RED, 0.30)
    if bbox is not None:
        draw_face_box(frame, bbox, RED, "Desconhecido", score)
    banner(frame, "ACESSO NEGADO", message, RED)
    return frame


def render_liveness(frame, bbox, name, score, ear, blinks, required, threshold):
    tint(frame, AMBER, 0.22)
    if bbox is not None:
        draw_face_box(frame, bbox, AMBER, name, score)
    banner(frame, f"OLA, {name.upper()}", "Prova de vida: PISQUE para confirmar", AMBER)
    draw_ear_bar(frame, ear, threshold, blinks, required)
    return frame


def render_granted(frame, name):
    tint(frame, GREEN, 0.30)
    draw_lock(frame, (frame.shape[1] // 2, frame.shape[0] // 2 - 40), 70, False, WHITE)
    banner(frame, "ACESSO PERMITIDO", f"Bem-vindo, {name}!", GREEN)
    return frame


def render_warning(frame, message):
    """Aviso de qualidade (ex.: 'Aproxime-se', 'Iluminação ruim') sobre fundo vermelho."""
    tint(frame, RED, 0.30)
    banner(frame, "SISTEMA BLOQUEADO", message, RED)
    return frame
