"""
Dataset - LIBRAS NN Platform
-------------------------------
Carrega os pools brutos JÁ existentes (gerados pelo capture_signatures.py
do projeto tradutor-libras) e monta datasets PyTorch prontos pra treino:

    - StaticSignDataset  -> letras estáticas (data_raw.json) + comandos
                            (commands_raw.json), combinados numa única
                            lista de classes (já que ambos são pose única
                            de 21 landmarks e vão pra mesma rede MLP).
    - MovementSignDataset -> letras de movimento (movement_raw.json),
                              sequências (T, 21, 3) já reamostradas pro
                              mesmo tamanho fixo, pra rede LSTM.

Fase atual: só lê data_raw.json / commands_raw.json / movement_raw.json.
A integração com a pasta imgs/ (preset de fotos) fica pra depois -- ver
comentário marcado com TODO(imgs) mais abaixo, ponto de extensão já
deixado pronto.

Formato esperado dos raw JSONs (igual ao que capture_signatures.py já
gera hoje):
    data_raw.json / commands_raw.json:
        { "<label>": [ [[x,y,z]*21], [[x,y,z]*21], ... ] }   # N amostras
    movement_raw.json:
        { "<label>": [ [[[x,y,z]*21] * T], ... ] }            # N sequências de T frames

Uso básico:
    from dataset import StaticSignDataset, MovementSignDataset, LabelEncoder

    static_ds = StaticSignDataset("data_raw.json", "commands_raw.json")
    train_ds, val_ds = static_ds.split(val_ratio=0.15)

    movement_ds = MovementSignDataset("movement_raw.json")
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Encoder de rótulos (classe <-> índice), compartilhado entre datasets
# --------------------------------------------------------------------------

class LabelEncoder:
    """Mapeamento simples nome-da-classe <-> índice inteiro, serializável."""

    def __init__(self, labels: Optional[List[str]] = None):
        self.classes = sorted(set(labels)) if labels else []
        self._to_idx = {name: i for i, name in enumerate(self.classes)}

    def fit(self, labels: List[str]) -> "LabelEncoder":
        self.classes = sorted(set(labels))
        self._to_idx = {name: i for i, name in enumerate(self.classes)}
        return self

    def encode(self, label: str) -> int:
        return self._to_idx[label]

    def decode(self, idx: int) -> str:
        return self.classes[idx]

    def __len__(self) -> int:
        return len(self.classes)

    def to_json(self) -> Dict:
        return {"classes": self.classes}

    @classmethod
    def from_json(cls, data: Dict) -> "LabelEncoder":
        return cls(data["classes"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))


# --------------------------------------------------------------------------
# Utilidades de leitura dos raw JSONs
# --------------------------------------------------------------------------

def _load_raw_pool(path: str) -> Dict[str, list]:
    if not os.path.exists(path):
        print(f"Aviso: {path} não encontrado -- nenhuma amostra virá dessa fonte.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # normaliza: cada entrada pode ser uma amostra única (ndim variável) ou
    # uma lista de amostras -- garante sempre "lista de amostras".
    pool = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.ndim == 2:          # (21, 3) -> pose única -> vira lista com 1
            pool[name] = [arr]
        elif arr.ndim == 3 and arr.shape[1:] == (21, 3):
            # ambíguo entre "N poses estáticas (N,21,3)" e "1 sequência
            # (T,21,3)" -- quem chama decide via `kind` (ver loaders abaixo).
            pool[name] = [arr[i] for i in range(arr.shape[0])]
        else:
            pool[name] = arr  # já deve ser (N, T, 21, 3) -- movimento
    return pool


def _load_static_pool(path: str) -> Dict[str, List[np.ndarray]]:
    """Letras/comandos: cada entrada do pool é uma pose (21,3)."""
    if not os.path.exists(path):
        print(f"Aviso: {path} não encontrado -- nenhuma amostra virá dessa fonte.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pool = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.ndim == 2:
            pool[name] = [arr]
        else:  # (N, 21, 3)
            pool[name] = [arr[i] for i in range(arr.shape[0])]
    return pool


def _load_movement_pool(path: str) -> Dict[str, List[np.ndarray]]:
    """Movimento: cada entrada do pool é uma sequência (T,21,3)."""
    if not os.path.exists(path):
        print(f"Aviso: {path} não encontrado -- nenhuma amostra virá dessa fonte.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pool = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.ndim == 3:  # uma única sequência (T,21,3) capturada até agora
            pool[name] = [arr]
        else:  # (N, T, 21, 3)
            pool[name] = [arr[i] for i in range(arr.shape[0])]
    return pool


# --------------------------------------------------------------------------
# StaticSignDataset -- letras estáticas + comandos (mesma rede MLP)
# --------------------------------------------------------------------------

class StaticSignDataset(Dataset):
    """
    Combina data_raw.json (letras estáticas) e commands_raw.json (comandos)
    num único dataset de classificação multi-classe: cada amostra é um
    vetor de 63 floats (21 landmarks * xyz, já normalizados pelo pipeline
    de captura), e o rótulo é o nome da letra OU do comando.

    Opcionalmente, também engorda o pool com fotos da pasta imgs/ (preset
    de imagens por letra) via image_landmarks.extract_static_pool_from_imgs
    -- passe `imgs_dir` pra ativar. Sem `imgs_dir`, o comportamento é
    idêntico a antes (só data_raw_path + commands_raw_path).
    """

    def __init__(
        self,
        data_raw_path: str,
        commands_raw_path: Optional[str] = None,
        label_encoder: Optional[LabelEncoder] = None,
        imgs_dir: Optional[str] = None,
        imgs_model_path: Optional[str] = None,
    ):
        letters_pool = _load_static_pool(data_raw_path)
        commands_pool = _load_static_pool(commands_raw_path) if commands_raw_path else {}

        self.samples: List[np.ndarray] = []
        self.labels: List[str] = []
        for pool in (letters_pool, commands_pool):
            for name, poses in pool.items():
                for pose in poses:
                    self.samples.append(pose.reshape(-1).astype(np.float32))  # (63,)
                    self.labels.append(name)

        self.imgs_stats = None
        if imgs_dir:
            from image_landmarks import extract_static_pool_from_imgs, DEFAULT_MODEL_PATH

            imgs_pool, stats = extract_static_pool_from_imgs(
                imgs_dir, model_path=imgs_model_path or DEFAULT_MODEL_PATH,
            )
            self.imgs_stats = stats
            for name, poses in imgs_pool.items():
                for pose in poses:
                    self.samples.append(pose.reshape(-1).astype(np.float32))
                    self.labels.append(name)
            print(f"  imgs/: +{stats['images_processed']} imagens processadas, "
                  f"{sum(stats['per_label'].values())} poses extraídas -> {stats['per_label']}")

        if not self.samples:
            raise ValueError(
                f"Nenhuma amostra encontrada em '{data_raw_path}', '{commands_raw_path}' "
                f"nem em '{imgs_dir}'. Capture dados com capture_signatures.py antes de treinar."
            )

        self.label_encoder = label_encoder or LabelEncoder(self.labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.samples[idx])
        y = torch.tensor(self.label_encoder.encode(self.labels[idx]), dtype=torch.long)
        return x, y

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def split(self, val_ratio: float = 0.15, seed: int = 42) -> Tuple["StaticSubset", "StaticSubset"]:
        return _stratified_split(self, val_ratio, seed)


class StaticSubset(Dataset):
    """Fatia de um StaticSignDataset (usada pelo split train/val)."""

    def __init__(self, parent: StaticSignDataset, indices: List[int]):
        self.parent = parent
        self.indices = indices
        self.label_encoder = parent.label_encoder

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.parent[self.indices[idx]]


def _stratified_split(dataset: StaticSignDataset, val_ratio: float, seed: int):
    """Split treino/val mantendo a proporção de classes (importante aqui
    porque o dataset costuma ser pequeno e desbalanceado entre letras)."""
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = {}
    for idx, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return StaticSubset(dataset, train_idx), StaticSubset(dataset, val_idx)


# --------------------------------------------------------------------------
# MovementSignDataset -- letras de movimento (rede LSTM)
# --------------------------------------------------------------------------

class MovementSignDataset(Dataset):
    """
    Carrega movement_raw.json: cada amostra é uma sequência (T, 21, 3) já
    reamostrada pro tamanho fixo (dynamic_signatures.DEFAULT_SEQUENCE_LENGTH
    no projeto original). O rótulo é a letra de movimento (H, J, K, X, Z...).
    """

    def __init__(self, movement_raw_path: str, label_encoder: Optional[LabelEncoder] = None):
        pool = _load_movement_pool(movement_raw_path)

        self.samples: List[np.ndarray] = []
        self.labels: List[str] = []
        seq_len = None
        for name, sequences in pool.items():
            for seq in sequences:
                if seq_len is None:
                    seq_len = seq.shape[0]
                elif seq.shape[0] != seq_len:
                    raise ValueError(
                        f"Sequências de tamanho inconsistente pra '{name}': "
                        f"esperado T={seq_len}, veio T={seq.shape[0]}. "
                        "Todas as amostras de movement_raw.json devem estar reamostradas "
                        "pro mesmo comprimento (isso já é feito pelo capture_signatures.py)."
                    )
                self.samples.append(seq.reshape(seq.shape[0], -1).astype(np.float32))  # (T, 63)
                self.labels.append(name)

        if not self.samples:
            raise ValueError(
                f"Nenhuma amostra de movimento encontrada em '{movement_raw_path}'. "
                "Capture gestos de movimento com capture_signatures.py antes de treinar."
            )

        self.sequence_length = seq_len
        self.label_encoder = label_encoder or LabelEncoder(self.labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.samples[idx])  # (T, 63)
        y = torch.tensor(self.label_encoder.encode(self.labels[idx]), dtype=torch.long)
        return x, y

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def split(self, val_ratio: float = 0.15, seed: int = 42):
        return _stratified_split(self, val_ratio, seed)  # type: ignore[arg-type]


if __name__ == "__main__":
    # Teste rápido/manual: ajuste os caminhos e rode direto pra conferir
    # se os JSONs do projeto tradutor-libras estão sendo lidos certo.
    import argparse

    parser = argparse.ArgumentParser(description="Teste manual dos loaders de dataset.")
    parser.add_argument("--data-raw", default="data_raw.json")
    parser.add_argument("--commands-raw", default="commands_raw.json")
    parser.add_argument("--movement-raw", default="movement_raw.json")
    args = parser.parse_args()

    try:
        static_ds = StaticSignDataset(args.data_raw, args.commands_raw)
        print(f"StaticSignDataset: {len(static_ds)} amostras, {len(static_ds.label_encoder)} classes")
        print("  classes:", static_ds.label_encoder.classes)
        print("  contagem:", static_ds.class_counts())
    except ValueError as e:
        print(f"StaticSignDataset: {e}")

    try:
        movement_ds = MovementSignDataset(args.movement_raw)
        print(f"\nMovementSignDataset: {len(movement_ds)} amostras, "
              f"{len(movement_ds.label_encoder)} classes, T={movement_ds.sequence_length}")
        print("  classes:", movement_ds.label_encoder.classes)
        print("  contagem:", movement_ds.class_counts())
    except ValueError as e:
        print(f"MovementSignDataset: {e}")
