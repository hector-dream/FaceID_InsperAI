"""
FaceID — Insper AI 2026.1

Pacote com o pipeline completo de reconhecimento facial:

    config       — parâmetros centrais
    capture      — Dia 1: captura/cadastro de rostos (webcam + MediaPipe)
    embeddings   — Dia 2: extração de embeddings (InsightFace/ArcFace)
    recognition  — Dia 3: matching por similaridade de cosseno
    liveness     — Dia 4: prova de vida (anti-spoofing por piscada)
    interface    — Dia 5: telas de bloqueio/liberação (OpenCV)
    app          — Dia 5: fluxo de autenticação (máquina de estados)
"""

__version__ = "1.0.0"
