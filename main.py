"""
FaceID — Insper AI 2026.1
Ponto de entrada com subcomandos para todo o pipeline.

Uso:
    python main.py capture [--name NOME]   # Dia 1: cadastrar rostos pela webcam
    python main.py enroll                  # Dia 2: gerar a base de embeddings
    python main.py run [--threshold 0.4]   # Dias 3-5: autenticação em tempo real
    python main.py doctor                  # diagnóstico do ambiente/dependências

Sem argumentos, executa o sistema de autenticação ('run').
"""

import argparse
import sys


def cmd_capture(args):
    from src.capture import run_capture
    run_capture(person_name=args.name)


def cmd_enroll(args):
    from src.embeddings import build_database
    build_database()


def cmd_run(args):
    from src.app import run_auth
    run_auth(threshold=args.threshold)


def cmd_doctor(args):
    """Verifica versões e disponibilidade das dependências (sem abrir a webcam)."""
    import importlib

    print("🩺 Diagnóstico do ambiente FaceID\n")
    print(f"Python: {sys.version.split()[0]}")

    checks = ["cv2", "numpy", "mediapipe", "insightface", "onnxruntime"]
    all_ok = True
    for mod in checks:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            print(f"  ✅ {mod:<12} {version}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"  ❌ {mod:<12} NÃO disponível ({exc})")

    # Estado dos artefatos do projeto
    from src import config
    print()
    print(f"Modelo de detecção (Dia 1): "
          f"{'✅' if config.MP_DETECTOR_MODEL.exists() else '❌ (será baixado no 1º uso)'}")
    print(f"Base de embeddings (Dia 2): "
          f"{'✅ ' + str(config.EMBEDDINGS_PATH) if config.EMBEDDINGS_PATH.exists() else '❌ (rode: python main.py enroll)'}")

    if config.DATASET_DIR.exists():
        people = [p.name for p in config.DATASET_DIR.iterdir() if p.is_dir()]
        print(f"Pessoas cadastradas: {', '.join(people) if people else '(nenhuma)'}")
    else:
        print("Pessoas cadastradas: (dataset ainda não criado)")

    print("\nResumo:", "tudo certo! 🎉" if all_ok else "faltam dependências (veja acima).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceid",
        description="Sistema de Face ID - Insper AI 2026.1",
    )
    sub = parser.add_subparsers(dest="command")

    p_cap = sub.add_parser("capture", help="Dia 1: cadastrar rostos pela webcam")
    p_cap.add_argument("--name", default=None, help="nome da pessoa a cadastrar")
    p_cap.set_defaults(func=cmd_capture)

    p_enr = sub.add_parser("enroll", help="Dia 2: gerar base de embeddings")
    p_enr.set_defaults(func=cmd_enroll)

    p_run = sub.add_parser("run", help="Dias 3-5: autenticação em tempo real")
    p_run.add_argument("--threshold", type=float, default=None,
                       help="limiar de similaridade de cosseno (0-1)")
    p_run.set_defaults(func=cmd_run)

    p_doc = sub.add_parser("doctor", help="diagnóstico do ambiente")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        # Sem subcomando: roda a autenticação por padrão.
        from src.app import run_auth
        run_auth()
        return

    args.func(args)


if __name__ == "__main__":
    main()
