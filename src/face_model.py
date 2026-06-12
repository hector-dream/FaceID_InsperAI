"""
Arquitetura da rede de embeddings faciais — substitui o InsightFace/ArcFace.

`FaceEmbeddingNet` é um backbone do torchvision (pré-treinado na ImageNet) com
a cabeça de classificação original trocada por uma camada linear que reduz
para `embedding_dim`. É essa rede que `src/embeddings.py` carrega em produção.

`FaceClassifier` só existe para o treino (`src/train.py`): embute a
`FaceEmbeddingNet` e adiciona uma camada de classes, treinada com softmax
(cross-entropy). Depois do treino, a camada de classes é descartada e ficamos
só com a `FaceEmbeddingNet` — o mesmo papel que o ArcFace tinha antes.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class FaceEmbeddingNet(nn.Module):
    """Backbone + camada linear final -> vetor de embedding (não normalizado)."""

    def __init__(self, embedding_dim: int = 128, backbone: str = "mobilenet_v2", pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone

        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone desconhecido: {backbone!r} (use 'mobilenet_v2' ou 'resnet18')")

        self.backbone = net
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.embedding(features)


class FaceClassifier(nn.Module):
    """Cabeça de classificação (usada só durante o treino)."""

    def __init__(self, embedding_net: FaceEmbeddingNet, num_classes: int):
        super().__init__()
        self.embedding_net = embedding_net
        self.classifier = nn.Linear(embedding_net.embedding_dim, num_classes)

    def forward(self, x):
        emb = self.embedding_net(x)
        logits = self.classifier(emb)
        return logits, emb
