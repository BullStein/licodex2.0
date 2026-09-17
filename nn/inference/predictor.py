"""
Predictor de Produção - LIBRAS NN Platform
------------------------------------------------
Carrega os checkpoints "_latest" (os que o train.py/train_daemon.py
promovem quando não regridem) e faz a inferência de verdade -- isso
substitui o papel do ai_judge.py (juiz via Ollama) no fluxo do
auto_validador_ia.py: em vez de um LLM julgar um ranking de distâncias
k-NN, a própria rede treinada já devolve a classe + confiança direto.

Compatibilidade proposital com o formato que auto_validador_ia.py já
espera de volta de judge_prediction() -- mesmas chaves no dict
("veredito", "confianca", "sugestao", "raciocinio") -- pra trocar de
Ollama pra rede neural exigir o mínimo de mudança no script de captura.
Diferença: aqui a ENTRADA é a pose/sequência normalizada direto (não um
ranking k-NN pré-calculado), já que a rede substitui o classificador
inteiro, não só a etapa de julgamento.

HOT-RELOAD: como o train_daemon.py roda em paralelo promovendo novos
checkpoints (_latest.pt) sem avisar ninguém, o predictor confere o
mtime desses arquivos a cada chamada de predict_* e recarrega sozinho
se detectar uma versão mais nova -- assim o tradutor em produção pega
o modelo mais recente sem precisar reiniciar o processo.

Uso básico:
    from predictor import SignPredictor

    predictor = SignPredictor(checkpoints_dir="../checkpoints")

    # pose: np.ndarray (21, 3) normalizada (mesmo formato do
    # normalize_landmarks() dos scripts do tradutor-libras)
    veredito = predictor.predict_static(pose)
    # veredito = {"veredito": "correto"|"incerto", "confianca": 0.0-1.0,
    #             "sugestao": "B", "raciocinio": "..."}

    # sequence: np.ndarray (T, 21, 3) já reamostrada
    veredito = predictor.predict_movement(sequence)
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

# Os modelos e o LabelEncoder vivem em train/ (mesmo projeto, pasta
# irmã) -- em vez de duplicar esses arquivos aqui em inference/, aponta
# o sys.path pra lá. Se você preferir, pode transformar o projeto num
# pacote de verdade (com __init__.py) e trocar isso por um import
# relativo -- deixei assim pra manter cada pasta rodável isoladamente.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))

from dataset import LabelEncoder      # noqa: E402
from model_static import StaticSignNet    # noqa: E402
from model_movement import MovementSignNet  # noqa: E402


class _LoadedModel:
    """Agrupa modelo + label encoder + mtime do checkpoint carregado, pra
    permitir hot-reload sem reimplementar a lógica em duplicidade."""

    def __init__(self, model: Optional[torch.nn.Module], label_encoder: Optional[LabelEncoder], mtime: float):
        self.model = model
        self.label_encoder = label_encoder
        self.mtime = mtime


def _checkpoint_paths(checkpoints_dir: str, prefix: str):
    ckpt = os.path.join(checkpoints_dir, f"{prefix}_latest.pt")
    labels = os.path.join(checkpoints_dir, f"{prefix}_latest_labels.json")
    return ckpt, labels


def _load_model(checkpoints_dir: str, prefix: str, model_cls, model_kwargs_fn) -> _LoadedModel:
    ckpt_path, labels_path = _checkpoint_paths(checkpoints_dir, prefix)
    if not os.path.exists(ckpt_path) or not os.path.exists(labels_path):
        return _LoadedModel(None, None, 0.0)

    label_encoder = LabelEncoder.load(labels_path)
    model = model_cls(**model_kwargs_fn(len(label_encoder)))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    mtime = os.path.getmtime(ckpt_path)
    return _LoadedModel(model, label_encoder, mtime)


class SignPredictor:
    """
    Ponto único de inferência em produção. Mantém os dois modelos
    (estático e movimento) carregados em memória e faz hot-reload
    automático quando o train_daemon.py promove um checkpoint novo.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        auto_confirm: float = 0.75,
        auto_correct: float = 0.80,
        auto_reload: bool = True,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.auto_confirm = auto_confirm
        self.auto_correct = auto_correct
        self.auto_reload = auto_reload

        self._static = self._reload_static()
        self._movement = self._reload_movement()

    # -- carregamento / hot-reload ---------------------------------------

    def _reload_static(self) -> _LoadedModel:
        return _load_model(
            self.checkpoints_dir, "model_static", StaticSignNet,
            lambda n: {"num_classes": n},
        )

    def _reload_movement(self) -> _LoadedModel:
        return _load_model(
            self.checkpoints_dir, "model_movement", MovementSignNet,
            lambda n: {"num_classes": n},
        )

    def _maybe_reload_static(self):
        if not self.auto_reload:
            return
        ckpt_path, _ = _checkpoint_paths(self.checkpoints_dir, "model_static")
        if os.path.exists(ckpt_path):
            mtime = os.path.getmtime(ckpt_path)
            if mtime != self._static.mtime:
                print(f"[predictor] novo checkpoint estático detectado (mtime mudou), recarregando...")
                self._static = self._reload_static()

    def _maybe_reload_movement(self):
        if not self.auto_reload:
            return
        ckpt_path, _ = _checkpoint_paths(self.checkpoints_dir, "model_movement")
        if os.path.exists(ckpt_path):
            mtime = os.path.getmtime(ckpt_path)
            if mtime != self._movement.mtime:
                print(f"[predictor] novo checkpoint de movimento detectado (mtime mudou), recarregando...")
                self._movement = self._reload_movement()

    @property
    def static_ready(self) -> bool:
        return self._static.model is not None

    @property
    def movement_ready(self) -> bool:
        return self._movement.model is not None

    # -- inferência --------------------------------------------------------

    def _predict_common(self, loaded: _LoadedModel, x: torch.Tensor) -> Dict:
        if loaded.model is None:
            return {
                "veredito": "indisponivel", "confianca": 0.0, "sugestao": None,
                "raciocinio": "Nenhum checkpoint em produção ainda -- rede não foi treinada/promovida.",
            }

        with torch.no_grad():
            logits = loaded.model(x.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            top2 = torch.topk(probs, k=min(2, probs.shape[0]))
            top1_prob = top2.values[0].item()
            top1_idx = top2.indices[0].item()
            top2_prob = top2.values[1].item() if top2.values.shape[0] > 1 else 0.0

        sugestao = loaded.label_encoder.decode(top1_idx)
        margem = top1_prob - top2_prob

        # Diferente do ai_judge.py (que comparava contra um ranking k-NN e
        # podia discordar do 1º colocado), aqui só existe o top1 da própria
        # rede -- não há "errado" possível, só "confiável o bastante" ou não.
        veredito = "correto" if top1_prob >= self.auto_confirm else "incerto"

        return {
            "veredito": veredito,
            "confianca": round(top1_prob, 4),
            "sugestao": sugestao,
            "raciocinio": f"Rede prevê '{sugestao}' com confiança {top1_prob:.3f} "
                          f"(margem pro 2º colocado: {margem:.3f}).",
        }

    def predict_static(self, normalized_pose: np.ndarray) -> Dict:
        """normalized_pose: (21, 3) -- letra estática OU comando (mesma
        rede cobre os dois, já que ambos treinam juntos)."""
        self._maybe_reload_static()
        x = torch.from_numpy(np.asarray(normalized_pose, dtype=np.float32).reshape(-1))  # (63,)
        return self._predict_common(self._static, x)

    def predict_movement(self, sequence: np.ndarray) -> Dict:
        """sequence: (T, 21, 3) já reamostrada pro tamanho fixo de treino."""
        self._maybe_reload_movement()
        arr = np.asarray(sequence, dtype=np.float32)
        x = torch.from_numpy(arr.reshape(arr.shape[0], -1))  # (T, 63)
        return self._predict_common(self._movement, x)


if __name__ == "__main__":
    # Smoke test: usa os checkpoints de /tmp/checkpoints se existirem
    # (gerados pelo train.py), só pra validar que o carregamento e a
    # inferência funcionam de ponta a ponta.
    import argparse

    parser = argparse.ArgumentParser(description="Teste manual do SignPredictor.")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    args = parser.parse_args()

    predictor = SignPredictor(args.checkpoints_dir)
    print(f"static_ready={predictor.static_ready}  movement_ready={predictor.movement_ready}")

    if predictor.static_ready:
        fake_pose = np.random.uniform(-1, 1, size=(21, 3))
        print("predict_static:", predictor.predict_static(fake_pose))

    if predictor.movement_ready:
        fake_seq = np.random.uniform(-1, 1, size=(15, 21, 3))
        print("predict_movement:", predictor.predict_movement(fake_seq))
