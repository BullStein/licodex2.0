"""
Modelo de Movimento - LIBRAS NN Platform
--------------------------------------------
LSTM bidirecional pra classificar letras de movimento (H, J, K, X, Z...)
a partir de uma sequência de poses já reamostrada pro tamanho fixo
(mesmo SEQUENCE_LENGTH do dynamic_signatures.py original).

Entrada: (B, T, 63) -- T frames, 63 = 21 landmarks * xyz por frame.
Saída: logits (B, num_classes), igual ao StaticSignNet -- softmax fica
por conta de quem chama (treino usa CrossEntropyLoss direto nos logits;
inferência aplica softmax na hora de calcular a confiança).

Por que bidirecional: o gesto inteiro já está disponível de uma vez no
momento da classificação (a gravação já terminou quando classificamos),
então não há motivo pra abrir mão do contexto "futuro" dentro da própria
sequência -- diferente de um cenário de streaming em tempo real, onde só
a direção "passado -> presente" faria sentido.

Uso básico:
    from model_movement import MovementSignNet

    model = MovementSignNet(num_classes=len(label_encoder))
    logits = model(batch_x)              # batch_x: (B, T, 63)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_idx = probs.max(dim=1)
"""

import torch
import torch.nn as nn

INPUT_DIM = 63  # 21 landmarks * (x, y, z) por frame


class MovementSignNet(nn.Module):
    """
    LSTM bidirecional -> pega o último estado oculto de cada direção,
    concatena, passa por uma cabeça densa com dropout. Dataset de
    movimento tende a ser bem menor que o de letras estáticas (cada
    amostra é uma gravação inteira, não uma foto instantânea), então o
    hidden_dim começa modesto pra evitar overfitting muito rápido.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 63) -> logits: (B, num_classes)."""
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_dim) -- pega só a
        # última camada, das duas direções, e concatena.
        if self.lstm.bidirectional:
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            final_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            final_hidden = h_n[-1]
        return self.head(self.dropout(final_hidden))

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Retorna (confiança, índice_da_classe) pro batch inteiro."""
        self.eval()
        probs = torch.softmax(self(x), dim=1)
        confidence, pred_idx = probs.max(dim=1)
        return confidence, pred_idx


if __name__ == "__main__":
    # Smoke test: forward pass com batch fake antes de plugar dataset real.
    model = MovementSignNet(num_classes=5)  # H, J, K, X, Z, exemplo
    model.eval()
    fake_batch = torch.randn(3, 20, INPUT_DIM)  # B=3, T=20
    out = model(fake_batch)
    print("Logits shape:", out.shape)
    conf, pred = model.predict(fake_batch)
    print("Confiança:", conf)
    print("Predição (índice):", pred)
