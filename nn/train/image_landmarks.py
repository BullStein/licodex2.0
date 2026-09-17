"""
Extração de Landmarks a partir de Imagens - LIBRAS NN Platform
--------------------------------------------------------------------
Ponto de integração da pasta imgs/ (preset de fotos por letra, do
projeto tradutor-libras) com o dataset da rede estática. Reaproveita a
MESMA lógica de detecção/normalização do batch_train_from_images.py
original, só que devolvendo os landmarks em memória (dict) em vez de
gravar em data_raw.json -- quem decide o que fazer com o resultado é
o dataset.py (StaticSignDataset).

Estrutura esperada da pasta imgs/ (igual ao projeto original):
    imgs/
      A/       foto1.jpg  foto2.png ...
      B/       ...
      CA/      (= Ç, sem cedilha no nome da pasta)
      BACKSPACE/  SPACE/  CLEAR/

Uso básico:
    from image_landmarks import extract_static_pool_from_imgs

    pool, stats = extract_static_pool_from_imgs("imgs/")
    # pool = {"A": [np.ndarray(21,3), ...], "BACKSPACE": [...], ...}
    # stats = {"images_processed": N, "skipped_no_hand": [...], ...}
"""

import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

WRIST = 0
MIDDLE_MCP = 9

LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)} | {"Ç", "CA"}
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: str = DEFAULT_MODEL_PATH) -> None:
    if not os.path.exists(model_path):
        print("Baixando modelo hand_landmarker.task (primeira execução com imgs/)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download concluído:", model_path)


def normalize_landmarks(landmarks) -> np.ndarray:
    """Mesma normalização usada em todo o projeto tradutor-libras: origem
    no pulso, escala pela distância pulso -> base do dedo médio."""
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array(
        [[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale] for lm in landmarks],
        dtype=np.float64,
    )


def resolve_label(folder_name: str) -> str:
    """Mapeia o nome da subpasta pro rótulo usado no treino, ou None se
    não reconhecido (mesma regra do batch_train_from_images.py: 'CA' vira 'Ç')."""
    name = folder_name.strip().upper()
    if name in COMMANDS:
        return name
    if name == "CA":
        return "Ç"
    if name in LETTERS:
        return name
    return None


def extract_static_pool_from_imgs(
    imgs_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    min_detection_confidence: float = 0.5,
) -> Tuple[Dict[str, List[np.ndarray]], dict]:
    """
    Roda o MediaPipe HandLandmarker (modo IMAGEM) sobre todas as fotos de
    imgs/<label>/*.{jpg,png,...} e devolve:
        pool  -> {label: [pose (21,3), ...]}
        stats -> {"images_processed": N, "skipped_no_hand": [...],
                   "unknown_folders": [...], "per_label": {label: n}}

    Se imgs_dir não existir, devolve pool vazio e um aviso em stats --
    não é erro fatal, só significa "sem fotos extras pra essa rodada".
    """
    stats = {"images_processed": 0, "skipped_no_hand": [], "unknown_folders": [], "per_label": {}}

    if not imgs_dir or not os.path.isdir(imgs_dir):
        stats["warning"] = f"Pasta '{imgs_dir}' não encontrada -- pulando integração com imgs/."
        print(stats["warning"])
        return {}, stats

    # Import tardio: mediapipe é pesado pra carregar e só é necessário
    # quando de fato há uma pasta imgs/ pra processar.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ensure_model(model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    pool: Dict[str, List[np.ndarray]] = defaultdict(list)

    subfolders = sorted(
        d for d in os.listdir(imgs_dir) if os.path.isdir(os.path.join(imgs_dir, d))
    )
    for folder in subfolders:
        label = resolve_label(folder)
        folder_path = os.path.join(imgs_dir, folder)
        images = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)
        )
        if label is None:
            if images:
                stats["unknown_folders"].append(folder)
            continue

        for fname in images:
            stats["images_processed"] += 1
            fpath = os.path.join(folder_path, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                stats["skipped_no_hand"].append(f"{folder}/{fname} (falha ao ler imagem)")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                stats["skipped_no_hand"].append(f"{folder}/{fname}")
                continue

            # mesma regra do batch_train_from_images.py: se mais de uma
            # mão for detectada, usa a de maior confiança.
            best_idx = 0
            if len(result.handedness) > 1:
                best_idx = max(range(len(result.handedness)), key=lambda i: result.handedness[i][0].score)

            pose = normalize_landmarks(result.hand_landmarks[best_idx])
            pool[label].append(pose)

    landmarker.close()

    stats["per_label"] = {label: len(poses) for label, poses in pool.items()}
    return dict(pool), stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teste manual da extração de landmarks a partir de imgs/.")
    parser.add_argument("--imgs-dir", default="imgs")
    args = parser.parse_args()

    pool, stats = extract_static_pool_from_imgs(args.imgs_dir)
    print(f"\nImagens processadas: {stats['images_processed']}")
    print(f"Por label: {stats['per_label']}")
    if stats["skipped_no_hand"]:
        print(f"Sem mão detectada ({len(stats['skipped_no_hand'])}): {stats['skipped_no_hand'][:10]}")
    if stats["unknown_folders"]:
        print(f"Pastas não reconhecidas: {stats['unknown_folders']}")
