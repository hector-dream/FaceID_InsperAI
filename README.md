# Face ID

Projeto final da Insper AI - 20261

## Descrição

Sistema de reconhecimento e detecção facial desenvolvido como trabalho final. O projeto implementa um pipeline completo de captura de imagens de rostos e detecção de características faciais utilizando modelos otimizados em TensorFlow Lite.

## Equipe

- Hector Mathias
- Lara
- Anderson

## Estrutura do Projeto

```
src/
  __init__.py       - Inicialização do pacote
  capture.py        - Módulo de captura de imagens via webcam

data/
  dataset/          - Diretório para armazenamento de dados capturados

blaze_face_short_range.tflite  - Modelo pré-treinado para detecção facial
```

## Requisitos

- Python 3.8+
- MediaPipe
- OpenCV
- TensorFlow Lite

## Instalação

1. Clone ou faça download do repositório
2. Instale as dependências:

```bash
uv install
```

ou

```bash
pip install -r requirements.txt
```
