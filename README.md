# Face ID — Insper AI 2026.1

Sistema de **reconhecimento facial com prova de vida** (anti-spoofing) para o projeto final do trainee. Simula um Face ID real:
a tela começa **bloqueada (vermelha)**, reconhece o rosto cadastrado, exige uma
**piscada** para provar que é uma pessoa de verdade (e não uma foto) e então
**libera o acesso (verde)**.

## Equipe

- Hector Mathias
- Lara
- Anderson

> **Nota**: Execute `uv sync` para instalar as dependências antes da primeira execução.

---

## Pipeline

| Etapa | Módulo | Tecnologia |
|-------|--------|------------|
| Treino da rede de embeddings | `src/train.py`, `src/face_model.py` | PyTorch (MobileNetV2 + LFW) |
| Captura e cadastro de rostos pela webcam | `src/capture.py` | OpenCV + MediaPipe FaceDetector |
| Extração de *embeddings* (vetores 128-d) | `src/embeddings.py` | CNN própria (MobileNetV2 treinada no LFW) |
| Reconhecimento / *matching* | `src/recognition.py` | Similaridade de cosseno |
| Prova de vida (anti-spoofing por piscada) | `src/liveness.py` | MediaPipe FaceMesh (EAR) |
| nterface de bloqueio + fluxo de autenticação | `src/interface.py`, `src/app.py` | OpenCV |

Todos os parâmetros ajustáveis (limiares, câmera, modelo) ficam em **`src/config.py`**.

```
src/
  config.py        # parâmetros centrais (limiares, caminhos, câmera, treino)
  face_model.py    # — arquitetura da CNN de embeddings (PyTorch)
  train.py         # — treino da CNN no dataset LFW
  capture.py       # — captura/cadastro
  embeddings.py    # — detecção (MediaPipe) + embeddings (CNN própria)
  recognition.py   # — matching por cosseno
  liveness.py      # — prova de vida (piscada/EAR)
  interface.py     # — desenho das telas (bloqueado/liberado)
  app.py           # — máquina de estados da autenticação
data/
  dataset/<pessoa>/*.jpg   # fotos cadastradas
  models/face_embedding.pt # pesos da CNN treinada (gerado por `train`)
  embeddings.pkl           # base de vetores
main.py            # CLI: train / capture / enroll / run / doctor
```

---

## Como funciona o reconhecimento

0. **Treino**: uma CNN (MobileNetV2, pré-treinada na ImageNet) é treinada do
   zero para classificar as pessoas do dataset **LFW** (Labeled Faces in the
   Wild). Depois do treino, descartamos a camada de classes e ficamos só com
   a penúltima camada — ela passa a gerar o **embedding** do rosto. É o mesmo
   princípio do FaceNet/ArcFace, só que treinado por nós (`uv run main.py train`).
1. Essa rede transforma cada rosto em um vetor de **128 números** (*embedding*).
   Rostos parecidos geram vetores parecidos.
2. No cadastro, guardamos os vetores das suas fotos em `data/embeddings.pkl`.
3. Ao vivo, comparamos o vetor do rosto na câmera com a base usando
   **similaridade de cosseno**. Se a maior similaridade passar do limiar
   (`RECOGNITION_THRESHOLD`, padrão `0.75` — recalibrado empiricamente após o
   treino padrão, veja "Ajuste fino"), é você; senão, "Desconhecido".
4. A **prova de vida** calcula o *Eye Aspect Ratio* (EAR) com os marcos
   faciais do MediaPipe e detecta uma **piscada**. Uma foto estática não pisca,
   então o acesso só é liberado depois da piscada.

---

## Instalação

> **Recomendado: Python 3.11.** O PyTorch e o MediaPipe têm wheels mais
> estáveis em 3.10–3.12. O projeto foi ajustado para essa faixa.

### Com `uv`

```bash
# Cria o ambiente com Python 3.11
uv venv --python 3.11
# Ative o ambiente:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Linux/Mac:             source .venv/bin/activate

uv sync

#ou

uv pip install -r requirements.txt
```

### Com `pip`

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### ⚠️ PyTorch no Windows

`pip install torch` "puro" no Windows baixa uma build com CUDA embutido (vários
GB). Para a versão **CPU** (~200MB, suficiente para este projeto), instale com:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Quem usa `uv sync` com o `pyproject.toml` deste repo já recebe a build CPU
automaticamente (configurado em `[tool.uv.sources]`).

### Conferir o ambiente

```bash
uv run main.py doctor
```

Mostra as versões instaladas, se os modelos (detecção e embeddings) existem e
quem já está cadastrado.

---

## Uso

### 0) Treinar o modelo de embeddings

```bash
uv run main.py train
```

Baixa o dataset **LFW** (na primeira vez, ~200MB, fica em cache em
`~/scikit_learn_data/`) e treina a CNN (`src/face_model.py`) por algumas
épocas. Ao final salva os pesos em `data/models/face_embedding.pt` — é esse
arquivo que `src/embeddings.py` carrega para gerar os embeddings dos rostos.

Parâmetros opcionais (veja `src/config.py` para os padrões):

```bash
uv run main.py train --min-faces 10 --epochs 12 --batch-size 32 --lr 1e-3
```

- `--min-faces`: mínimo de fotos por pessoa no LFW. Menor = mais pessoas/classes
  (melhor generalização para rostos novos), porém treino mais lento.
- `--epochs`: mais épocas tendem a melhorar a separação dos embeddings, mas
  demoram mais em CPU.

Só é preciso treinar de novo se quiser ajustar a arquitetura/hiperparâmetros —
o checkpoint gerado já fica pronto para os passos seguintes.

### 1) Cadastrar um rosto

```bash
uv run main.py capture --name <SEUNOME>
```

Aponte o rosto para a câmera e aperte **`C`** para salvar (capture de **5 a 10
fotos** em ângulos/expressões levemente diferentes). Aperte **`Q`** para sair.
As fotos vão para `data/dataset/<SEUNOME>/`. Repita para cada pessoa.

### 2) Gerar a base de embeddings

```bash
uv run main.py enroll
```

Lê todas as fotos de `data/dataset/`, gera os vetores com a CNN treinada e
salva em `data/embeddings.pkl`. Rode novamente sempre que adicionar/remover
fotos **ou treinar o modelo de novo** (os embeddings antigos ficam
incompatíveis com um novo checkpoint).

### 3) Rodar a autenticação

```bash
uv run main.py run
# limiar mais permissivo (menos falsos negativos, p.ex. se não estiver te
# reconhecendo) ou mais rígido (menos falsos positivos):
uv run main.py run --threshold 0.65
uv run main.py run --threshold 0.85
```

Fluxo na tela: **BLOQUEADO** → rosto reconhecido → **PISQUE** (prova de vida) →
**ACESSO PERMITIDO**. Controles: **`F`** alterna tela cheia, **`Q`/`ESC`** sai.

---

## Ajuste fino

Edite `src/config.py`:

| Parâmetro | O que faz | Dica |
|-----------|-----------|------|
| `RECOGNITION_THRESHOLD` | Rigor do reconhecimento (cosseno 0–1) | Subir reduz falsos positivos; baixar reduz falsos negativos. **Recalibre após treinar/`enroll`** — a escala das similaridades muda a cada modelo treinado. |
| `EAR_THRESHOLD` | Sensibilidade da piscada | Subir detecta piscadas mais leves |
| `REQUIRED_BLINKS` | Nº de piscadas para liberar | 1 é suficiente para barrar foto |
| `LIVENESS_TIMEOUT_S` | Tempo para piscar | — |
| `MIN_FACE_AREA_RATIO` | Distância mínima do rosto | Aciona "Aproxime-se da câmera" |
| `MIN_BRIGHTNESS` | Luz mínima aceitável | Aciona "Iluminação insuficiente" |
| `LFW_MIN_FACES_PER_PERSON`, `TRAIN_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_LR` | Hiperparâmetros do `train` | Mais épocas / `min_faces` menor = melhor qualidade, treino mais lento |
| `EMBEDDING_BACKBONE` | `"mobilenet_v2"` (leve) ou `"resnet18"` (mais pesado) | Troque e retreine se quiser comparar |

> 💡 Para calibrar `RECOGNITION_THRESHOLD`, use
> `FaceRecognizer.identify_per_person()` (em `src/recognition.py`) para ver as
> similaridades de cada pessoa cadastrada e escolher um limiar que separe bem
> "você" de "outra pessoa".

---

## Tratamento de exceções

O sistema permanece bloqueado e mostra um aviso quando:

- **Nenhum rosto** detectado → "Nenhum rosto detectado"
- **Mais de um rosto** na cena → "Múltiplos rostos detectados" (segurança)
- **Rosto muito distante** → "Aproxime-se da câmera"
- **Iluminação ruim** → "Iluminação insuficiente"

---

## Solução de problemas

- **Webcam não abre:** ajuste `CAMERA_INDEX` em `src/config.py` (0, 1, 2...).
- **"Modelo de embeddings não encontrado":** rode `uv run main.py train` antes
  de `enroll`/`run`.
- **Está lento:** o `EMBEDDING_BACKBONE` padrão (`mobilenet_v2`) já é leve;
  reduzir `EMBEDDING_IMAGE_SIZE` em `src/config.py` (e retreinar) acelera ainda mais.
- **Não te reconhece:** capture mais fotos com boa luz e refaça o `enroll`; se
  necessário, baixe um pouco o `RECOGNITION_THRESHOLD`.
- **Reconhece foto sua:** é o anti-spoofing entrando em ação — a piscada é
  obrigatória. Para mais segurança, aumente `REQUIRED_BLINKS`.
