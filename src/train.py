"""
Treina a rede de embeddings faciais (`src/face_model.FaceEmbeddingNet`) no
dataset **LFW** (Labeled Faces in the Wild), tratando o problema como
**classificação** (softmax): a rede aprende a distinguir as pessoas do LFW.

Depois do treino, descartamos a camada de classes (`FaceClassifier.classifier`)
e guardamos só a `FaceEmbeddingNet` — o vetor que ela produz (penúltima
camada) passa a ser o "embedding" do rosto, no lugar do que o ArcFace/
InsightFace fazia antes.

Uso:
    uv run main.py train
    uv run main.py train --min-faces 10 --epochs 12
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config
from .face_model import FaceClassifier, FaceEmbeddingNet

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _LFWDataset(Dataset):
    """Embrulha o array de imagens do LFW (N, H, W, 3) em float [0, 1]."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = (self.images[idx] * 255).astype(np.uint8)
        return self.transform(image), int(self.labels[idx])


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.ToTensor(), transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))
    return transforms.Compose(ops)


@torch.no_grad()
def _evaluate(model: FaceClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train_model(
    min_faces_per_person: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    verbose: bool = True,
) -> None:
    """Baixa o LFW, treina a CNN de embeddings e salva os pesos em `config.EMBEDDING_MODEL_PATH`."""
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    min_faces_per_person = min_faces_per_person or config.LFW_MIN_FACES_PER_PERSON
    epochs = epochs or config.TRAIN_EPOCHS
    batch_size = batch_size or config.TRAIN_BATCH_SIZE
    lr = lr or config.TRAIN_LR

    if verbose:
        print(f"📦 Baixando/carregando o LFW (min_faces_per_person={min_faces_per_person})...")
        print("   (na primeira vez baixa ~200MB e fica em cache em ~/scikit_learn_data/)")

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.4, color=True, funneled=True)
    images = lfw.images  # (N, H, W, 3), float em [0, 1]
    labels = lfw.target
    num_classes = len(lfw.target_names)

    if verbose:
        print(f"   {len(images)} imagens | {num_classes} pessoas")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=config.TRAIN_VAL_SPLIT, stratify=labels, random_state=42
    )

    image_size = config.EMBEDDING_IMAGE_SIZE
    train_ds = _LFWDataset(X_train, y_train, _build_transform(image_size, train=True))
    val_ds = _LFWDataset(X_val, y_val, _build_transform(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    embedding_net = FaceEmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM, backbone=config.EMBEDDING_BACKBONE, pretrained=True
    )
    model = FaceClassifier(embedding_net, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"\n🧠 Treinando {config.EMBEDDING_BACKBONE} ({epochs} épocas, lote={batch_size}, lr={lr})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0

        for images_batch, labels_batch in train_loader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(images_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels_batch.size(0)
            running_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
            seen += labels_batch.size(0)

        train_loss = running_loss / seen
        train_acc = running_correct / seen
        val_acc = _evaluate(model, val_loader, device)

        if verbose:
            print(
                f"Época {epoch:2d}/{epochs} - loss: {train_loss:.4f} "
                f"- acc treino: {train_acc:.3f} - acc validação: {val_acc:.3f}"
            )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": embedding_net.state_dict(),
            "embedding_dim": config.EMBEDDING_DIM,
            "backbone": config.EMBEDDING_BACKBONE,
            "image_size": image_size,
        },
        config.EMBEDDING_MODEL_PATH,
    )

    if verbose:
        print(f"\n💾 Modelo de embeddings salvo em {config.EMBEDDING_MODEL_PATH}")
        print("   Rode 'uv run main.py enroll' para regerar a base com o novo modelo.")


if __name__ == "__main__":
    train_model()
