"""
Modelo Estático - LIBRAS NN Platform
----------------------------------------
MLP simples pra classificar letras estáticas + comandos a partir de uma
pose única (21 landmarks * xyz = 63 floats já normalizados).

Substitui o classify_ranked_weighted (k-NN por distância euclidiana
ponderada) do projeto tradutor-libras original por uma rede treinada de
verdade, mantendo a mesma entrada (pose normalizada) e trocando a saída:
em vez de um ranking por distância, um softmax com probabilidade por classe
-- o que também substitui o papel do juiz de IA via Ollama (ai_judge.py),
já que a confiança agora vem direto da rede.

Uso básico:
    from model_static import StaticSignNet

    model = StaticSignNet(num_classes=len(label_encoder))
    logits = model(batch_x)              # batch_x: (B, 63)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_idx = probs.max(dim=1)
"""

import torch
import torch.nn as nn

INPUT_DIM = 63  # 21 landmarks * (x, y, z)


class StaticSignNet(nn.Module):
    """
    MLP com 3 camadas ocultas + dropout (dataset costuma ser pequeno,
    então dropout ajuda a não decorar as amostras de treino) e
    normalização por batch pra estabilizar o treino incremental
    (o daemon vai fazer fine-tuning contínuo, não só um treino único).
    """

    def __init__(self, num_classes: int, hidden_dims=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 63) -> logits: (B, num_classes) (sem softmax -- aplique
        na hora da inferência ou use nn.CrossEntropyLoss no treino, que já
        espera logits crus)."""
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Retorna (confiança, índice_da_classe) pro batch inteiro."""
        self.eval()
        probs = torch.softmax(self(x), dim=1)
        confidence, pred_idx = probs.max(dim=1)
        return confidence, pred_idx


if __name__ == "__main__":
    # Smoke test: garante que um forward pass roda sem erro com um batch
    # fake, antes de plugar dataset de verdade.
    model = StaticSignNet(num_classes=29)  # 26 letras + Ç + 2 comandos, exemplo
    model.eval()
    fake_batch = torch.randn(4, INPUT_DIM)
    out = model(fake_batch)
    print("Logits shape:", out.shape)
    conf, pred = model.predict(fake_batch)
    print("Confiança:", conf)
    print("Predição (índice):", pred)
