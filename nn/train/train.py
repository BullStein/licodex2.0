"""
Treino - LIBRAS NN Platform
-------------------------------
Treina StaticSignNet (letras + comandos) e MovementSignNet (letras de
movimento), a partir dos raw JSONs do projeto tradutor-libras. Salva
checkpoints VERSIONADOS em checkpoints/ (model_static_vN.pt,
model_movement_vN.pt) e só promove um checkpoint pra "produção"
(model_static_latest.pt / model_movement_latest.pt, que é o que o
inference/predictor.py carrega) se a acurácia de validação não
regredir em relação à produção atual.

Isso é o núcleo que o train_daemon.py (próxima etapa) vai chamar toda
vez que detectar mudança nos raw JSONs -- ele reusa exatamente essa
função `run_training_cycle`, só troca "rodar uma vez via CLI" por
"rodar sempre que o watchdog disparar".

Uso básico:
    python train.py --data-raw data_raw.json --commands-raw commands_raw.json \
                     --movement-raw movement_raw.json --checkpoints-dir checkpoints

    python train.py --epochs 100 --lr 0.001 --skip-movement
    (--skip-movement / --skip-static pra treinar só uma das redes, útil
    se você só capturou dado novo de um dos dois tipos)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import StaticSignDataset, MovementSignDataset, LabelEncoder
from model_static import StaticSignNet
from model_movement import MovementSignNet


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 16
    val_ratio: float = 0.15
    patience: int = 10  # early stopping: para se val_loss não melhora por N épocas
    seed: int = 42


@dataclass
class TrainResult:
    kind: str                       # "static" ou "movement"
    val_accuracy: float
    val_loss: float
    epochs_ran: int
    num_classes: int
    num_samples: int
    checkpoint_path: str
    label_encoder_path: str
    promoted: bool


def _train_one_model(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg: TrainConfig,
    kind: str,
) -> tuple:
    """Loop de treino genérico (serve tanto pra StaticSignNet quanto
    MovementSignNet -- a diferença já está encapsulada no shape dos
    tensores que cada Dataset produz)."""
    torch.manual_seed(cfg.seed)

    # batch_size efetivo nunca maior que o dataset, e sempre >=2 pro
    # BatchNorm da rede estática não quebrar com batch de 1 amostra.
    train_batch = max(2, min(cfg.batch_size, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, drop_last=len(train_ds) > train_batch)
    val_batch = max(1, min(cfg.batch_size, len(val_ds))) if len(val_ds) > 0 else 1
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False) if len(val_ds) > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0
    last_val_accuracy = 0.0
    last_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        epochs_ran = epoch
        model.train()
        for x, y in train_loader:
            if x.shape[0] < 2 and kind == "static":
                continue  # BatchNorm precisa de >1 amostra no batch
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x)
                    loss = loss_fn(out, y)
                    total_loss += loss.item() * x.shape[0]
                    correct += (out.argmax(dim=1) == y).sum().item()
                    total += x.shape[0]
            last_val_loss = total_loss / max(1, total)
            last_val_accuracy = correct / max(1, total)

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    print(f"  [{kind}] early stopping na época {epoch} "
                          f"(sem melhora em {cfg.patience} épocas)")
                    break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{kind}] época {epoch}/{cfg.epochs} -- "
                  f"val_loss={last_val_loss:.4f} val_acc={last_val_accuracy:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, last_val_accuracy, best_val_loss, epochs_ran


def _next_version(checkpoints_dir: str, prefix: str) -> int:
    if not os.path.isdir(checkpoints_dir):
        return 1
    existing = [
        f for f in os.listdir(checkpoints_dir)
        if f.startswith(prefix + "_v") and f.endswith(".pt")
    ]
    versions = []
    for f in existing:
        try:
            versions.append(int(f[len(prefix) + 2 : -3]))
        except ValueError:
            continue
    return (max(versions) + 1) if versions else 1


def _current_production_accuracy(checkpoints_dir: str, prefix: str) -> Optional[float]:
    meta_path = os.path.join(checkpoints_dir, f"{prefix}_latest.meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("val_accuracy")


def _save_checkpoint(
    model: nn.Module,
    label_encoder: LabelEncoder,
    checkpoints_dir: str,
    prefix: str,
    val_accuracy: float,
    val_loss: float,
    extra_meta: dict,
) -> TrainResult:
    os.makedirs(checkpoints_dir, exist_ok=True)
    version = _next_version(checkpoints_dir, prefix)
    ckpt_path = os.path.join(checkpoints_dir, f"{prefix}_v{version}.pt")
    encoder_path = os.path.join(checkpoints_dir, f"{prefix}_v{version}_labels.json")

    torch.save(model.state_dict(), ckpt_path)
    label_encoder.save(encoder_path)

    meta = {
        "version": version,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "num_classes": len(label_encoder),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra_meta,
    }
    with open(os.path.join(checkpoints_dir, f"{prefix}_v{version}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    promoted = False
    current_best = _current_production_accuracy(checkpoints_dir, prefix)
    if current_best is None or val_accuracy >= current_best:
        latest_ckpt = os.path.join(checkpoints_dir, f"{prefix}_latest.pt")
        latest_encoder = os.path.join(checkpoints_dir, f"{prefix}_latest_labels.json")
        latest_meta = os.path.join(checkpoints_dir, f"{prefix}_latest.meta.json")
        torch.save(model.state_dict(), latest_ckpt)
        label_encoder.save(latest_encoder)
        with open(latest_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        promoted = True
        print(f"  [{prefix}] v{version} PROMOVIDO pra produção "
              f"(val_acc={val_accuracy:.3f}, anterior={current_best})")
    else:
        print(f"  [{prefix}] v{version} treinado mas NÃO promovido "
              f"(val_acc={val_accuracy:.3f} < produção atual {current_best:.3f})")

    return TrainResult(
        kind=prefix,
        val_accuracy=val_accuracy,
        val_loss=val_loss,
        epochs_ran=extra_meta.get("epochs_ran", 0),
        num_classes=len(label_encoder),
        num_samples=extra_meta.get("num_samples", 0),
        checkpoint_path=ckpt_path,
        label_encoder_path=encoder_path,
        promoted=promoted,
    )


def train_static(
    data_raw: str,
    commands_raw: str,
    checkpoints_dir: str,
    cfg: TrainConfig,
    imgs_dir: Optional[str] = None,
) -> Optional[TrainResult]:
    print("\n=== Treinando rede ESTÁTICA (letras + comandos) ===")
    try:
        dataset = StaticSignDataset(data_raw, commands_raw, imgs_dir=imgs_dir)
    except ValueError as e:
        print(f"  Pulado: {e}")
        return None

    train_ds, val_ds = dataset.split(val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"  {len(dataset)} amostras totais, {len(dataset.label_encoder)} classes "
          f"({len(train_ds)} treino / {len(val_ds)} val)")

    model = StaticSignNet(num_classes=len(dataset.label_encoder))
    model, val_acc, val_loss, epochs_ran = _train_one_model(model, train_ds, val_ds, cfg, kind="static")

    return _save_checkpoint(
        model, dataset.label_encoder, checkpoints_dir, "model_static",
        val_acc, val_loss, {"epochs_ran": epochs_ran, "num_samples": len(dataset)},
    )


def train_movement(
    movement_raw: str,
    checkpoints_dir: str,
    cfg: TrainConfig,
) -> Optional[TrainResult]:
    print("\n=== Treinando rede de MOVIMENTO (letras dinâmicas) ===")
    try:
        dataset = MovementSignDataset(movement_raw)
    except ValueError as e:
        print(f"  Pulado: {e}")
        return None

    train_ds, val_ds = dataset.split(val_ratio=cfg.val_ratio, seed=cfg.seed)
    print(f"  {len(dataset)} amostras totais, {len(dataset.label_encoder)} classes, "
          f"T={dataset.sequence_length} ({len(train_ds)} treino / {len(val_ds)} val)")

    model = MovementSignNet(num_classes=len(dataset.label_encoder))
    model, val_acc, val_loss, epochs_ran = _train_one_model(model, train_ds, val_ds, cfg, kind="movement")

    return _save_checkpoint(
        model, dataset.label_encoder, checkpoints_dir, "model_movement",
        val_acc, val_loss,
        {"epochs_ran": epochs_ran, "num_samples": len(dataset), "sequence_length": dataset.sequence_length},
    )


def run_training_cycle(
    data_raw: str,
    commands_raw: str,
    movement_raw: str,
    checkpoints_dir: str,
    cfg: TrainConfig,
    skip_static: bool = False,
    skip_movement: bool = False,
    imgs_dir: Optional[str] = None,
) -> dict:
    """Ponto de entrada único, reusado pelo train_daemon.py: roda um ciclo
    completo de treino (estático + movimento) e devolve os resultados."""
    results = {}
    if not skip_static:
        results["static"] = train_static(data_raw, commands_raw, checkpoints_dir, cfg, imgs_dir=imgs_dir)
    if not skip_movement:
        results["movement"] = train_movement(movement_raw, checkpoints_dir, cfg)
    return results


def main():
    parser = argparse.ArgumentParser(description="Treina as redes estática e de movimento da LIBRAS NN Platform.")
    parser.add_argument("--data-raw", default="data_raw.json")
    parser.add_argument("--commands-raw", default="commands_raw.json")
    parser.add_argument("--movement-raw", default="movement_raw.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-movement", action="store_true")
    parser.add_argument("--imgs-dir", default=None,
                         help="Pasta imgs/ (preset de fotos por letra) pra engordar o dataset estático. "
                              "Omitido = não usa imgs/, comportamento igual a antes.")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_ratio=args.val_ratio, patience=args.patience,
    )

    results = run_training_cycle(
        args.data_raw, args.commands_raw, args.movement_raw, args.checkpoints_dir, cfg,
        skip_static=args.skip_static, skip_movement=args.skip_movement, imgs_dir=args.imgs_dir,
    )

    print("\n=== Resumo ===")
    for kind, result in results.items():
        if result is None:
            print(f"  {kind}: pulado (sem dados)")
        else:
            print(f"  {kind}: val_acc={result.val_accuracy:.3f} "
                  f"({result.num_classes} classes, {result.epochs_ran} épocas) "
                  f"-> {'PROMOVIDO' if result.promoted else 'não promovido'}")


if __name__ == "__main__":
    main()
