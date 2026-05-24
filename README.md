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
| Captura e cadastro de rostos pela webcam | `src/capture.py` | OpenCV + MediaPipe FaceDetector |
| Extração de *embeddings* (vetores 512-d) | `src/embeddings.py` | InsightFace (ArcFace) |
| Reconhecimento / *matching* | `src/recognition.py` | Similaridade de cosseno |
| Prova de vida (anti-spoofing por piscada) | `src/liveness.py` | MediaPipe FaceMesh (EAR) |
| nterface de bloqueio + fluxo de autenticação | `src/interface.py`, `src/app.py` | OpenCV |

Todos os parâmetros ajustáveis (limiares, câmera, modelo) ficam em **`src/config.py`**.

```
src/
  config.py        # parâmetros centrais (limiares, caminhos, câmera)
  capture.py       # — captura/cadastro
  embeddings.py    # — embeddings (InsightFace)
  recognition.py   # — matching por cosseno
  liveness.py      # — prova de vida (piscada/EAR)
  interface.py     # — desenho das telas (bloqueado/liberado)
  app.py           # — máquina de estados da autenticação
data/
  dataset/<pessoa>/*.jpg   # fotos cadastradas 
  embeddings.pkl           # base de vetores
main.py            # CLI: capture / enroll / run / doctor
```

---

## Como funciona o reconhecimento

1. O **InsightFace** transforma cada rosto em um vetor de **512 números** (*embedding*).
   Rostos parecidos geram vetores parecidos.
2. No cadastro, guardamos os vetores das suas fotos em `data/embeddings.pkl`.
3. Ao vivo, comparamos o vetor do rosto na câmera com a base usando
   **similaridade de cosseno**. Se a maior similaridade passar do limiar
   (`RECOGNITION_THRESHOLD`, padrão `0.40`), é você; senão, "Desconhecido".
4. A **prova de vida** calcula o *Eye Aspect Ratio* (EAR) com os marcos
   faciais do MediaPipe e detecta uma **piscada**. Uma foto estática não pisca,
   então o acesso só é liberado depois da piscada.

---

## Instalação

> **Recomendado: Python 3.11.** O InsightFace/onnxruntime e o MediaPipe têm
> wheels mais estáveis em 3.10–3.12. O projeto foi ajustado para essa faixa.

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

### ⚠️ InsightFace no Windows

O `insightface` pode precisar compilar uma extensão durante a instalação. Se der
erro de compilação:

1. Instale o **"Microsoft C++ Build Tools"** (Desktop development with C++) e tente de novo, **ou**
2. Instale uma *wheel* pré-compilada do `insightface` compatível com a sua versão do Python e depois `pip install -r requirements.txt`.

Na primeira execução, o InsightFace baixa automaticamente o pacote de modelos
`buffalo_l` (~300 MB) para `~/.insightface/`. Precisa de internet só nessa vez.

### Conferir o ambiente

```bash
uv run main.py doctor
```

Mostra as versões instaladas, se o modelo de detecção existe e quem já está cadastrado.

---

## Uso

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

Lê todas as fotos de `data/dataset/`, gera os vetores e salva em `data/embeddings.pkl`.
Rode novamente sempre que adicionar/remover fotos.

### 3) Rodar a autenticação

```bash
uv run main.py run
# limiar mais rígido (menos falsos positivos):
uv run main.py run --threshold 0.5
```

Fluxo na tela: **BLOQUEADO** → rosto reconhecido → **PISQUE** (prova de vida) →
**ACESSO PERMITIDO**. Controles: **`F`** alterna tela cheia, **`Q`/`ESC`** sai.

---

## Ajuste fino

Edite `src/config.py`:

| Parâmetro | O que faz | Dica |
|-----------|-----------|------|
| `RECOGNITION_THRESHOLD` | Rigor do reconhecimento (cosseno 0–1) | Subir reduz falsos positivos; baixar reduz falsos negativos |
| `EAR_THRESHOLD` | Sensibilidade da piscada | Subir detecta piscadas mais leves |
| `REQUIRED_BLINKS` | Nº de piscadas para liberar | 1 é suficiente para barrar foto |
| `LIVENESS_TIMEOUT_S` | Tempo para piscar | — |
| `MIN_FACE_AREA_RATIO` | Distância mínima do rosto | Aciona "Aproxime-se da câmera" |
| `MIN_BRIGHTNESS` | Luz mínima aceitável | Aciona "Iluminação insuficiente" |
| `INSIGHTFACE_MODEL_PACK` | `buffalo_l` (preciso) / `buffalo_s` (rápido) | Use `buffalo_s` em CPUs lentas |

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
- **Está lento:** troque `INSIGHTFACE_MODEL_PACK` para `"buffalo_s"`.
- **Não te reconhece:** capture mais fotos com boa luz e refaça o `enroll`; se
  necessário, baixe um pouco o `RECOGNITION_THRESHOLD`.
- **Reconhece foto sua:** é o anti-spoofing entrando em ação — a piscada é
  obrigatória. Para mais segurança, aumente `REQUIRED_BLINKS`.
