"""
Reconhecimento de gestos COM MOVIMENTO - LIBRAS
--------------------------------------------------
Extensao do matching_improved.py para letras que nao sao uma pose
estatica, e sim um MOVIMENTO da mao no ar (ex.: H, J, K, X, Z,
dependendo da variante do alfabeto que voce estiver usando -- ajuste
livremente a lista MOVEMENT_LETTERS nos scripts que importam este
modulo).

Em vez de comparar uma unica pose (21 landmarks) contra outra,
comparamos uma SEQUENCIA de poses ao longo do tempo (uma
"trajetoria") contra outra sequencia. Como duas capturas do mesmo
gesto quase nunca tem o mesmo numero de frames (depende de quao
rapido a pessoa fez o movimento), toda sequencia e primeiro
RE-AMOSTRADA para um numero fixo de frames (`target_len`) por
interpolacao linear -- assim uma sequencia de 40 frames e uma de 22
frames viram ambas, por exemplo, 15 "posicoes-chave" comparaveis
ponto a ponto.

A distancia entre duas sequencias re-amostradas e a media (ao longo
dos frames) da mesma distancia ponderada por landmark usada para
letras estaticas (matching_improved.LANDMARK_WEIGHTS) -- pontas dos
dedos continuam pesando mais que o pulso, so que agora quadro a
quadro.

Formato de armazenamento (movement.json / movement_raw.json), mesma
logica de "pool que so cresce + media recalculada" do
data.json/data_raw.json:

    movement.json (medias, uma sequencia de T frames por alvo):
        { "H": [[[x,y,z]*21], ...T frames...], ... }

    movement_raw.json (pool completo, lista de sequencias por alvo):
        { "H": [ [[[x,y,z]*21]*T], [[[x,y,z]*21]*T], ... ], ... }
"""

import os
import json

import numpy as np

from matching_improved import LANDMARK_WEIGHTS

DEFAULT_SEQUENCE_LENGTH = 15


# ---------------------------------------------------------------------
# Reamostragem e distancia entre sequencias
# ---------------------------------------------------------------------

def resample_sequence(frames, target_len=DEFAULT_SEQUENCE_LENGTH):
    """
    frames: array-like (N, 21, 3), N variavel (>= 1) entre capturas.
    Retorna (target_len, 21, 3), reamostrado por interpolacao linear
    no eixo do tempo (cada uma das 21*3 = 63 coordenadas interpolada
    independentemente ao longo dos N frames originais).
    """
    frames = np.asarray(frames, dtype=np.float64)
    n = frames.shape[0]
    if n == target_len:
        return frames
    if n <= 1:
        return np.repeat(frames.reshape(1, 21, 3), target_len, axis=0)

    old_idx = np.linspace(0.0, 1.0, n)
    new_idx = np.linspace(0.0, 1.0, target_len)
    flat = frames.reshape(n, -1)  # (n, 63)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(new_idx, old_idx, flat[:, c])
    return out.reshape(target_len, 21, 3)


def sequence_weighted_distance(seq_a, seq_b):
    """seq_a, seq_b: (T, 21, 3), MESMO T. Retorna a media (ao longo dos
    T frames) da distancia ponderada por landmark entre as duas."""
    diff = seq_a - seq_b                                       # (T, 21, 3)
    dist_per_point = np.linalg.norm(diff, axis=2)                # (T, 21)
    weighted = dist_per_point * LANDMARK_WEIGHTS.T                # (T, 21)
    per_frame = weighted.sum(axis=1) / LANDMARK_WEIGHTS.sum()     # (T,)
    return float(per_frame.mean())


def classify_sequence_ranked(query_frames, signatures, k=3):
    """
    query_frames: (N, 21, 3) -- sequencia crua capturada agora (N
    variavel; sera reamostrada para o T de cada alvo antes de comparar).
    signatures: dict nome -> ndarray (n_amostras, T, 21, 3).
    Retorna [(nome, distancia), ...] do mais parecido pro menos, com
    voto k-NN por classe (media das K amostras mais proximas) -- mesma
    ideia do classify_ranked_weighted (matching_improved.py), so que
    em sequencias.
    """
    results = []
    for name, seqs in signatures.items():
        if seqs.shape[0] == 0:
            continue
        target_len = seqs.shape[1]
        query_resampled = resample_sequence(query_frames, target_len)
        dists = np.array([
            sequence_weighted_distance(query_resampled, seqs[i])
            for i in range(seqs.shape[0])
        ])
        k_eff = min(k, len(dists))
        nearest_k = np.partition(dists, k_eff - 1)[:k_eff]
        results.append((name, float(nearest_k.mean())))
    results.sort(key=lambda x: x[1])
    return results


def is_confident(ranking, threshold, min_margin=0.04):
    """Mesma logica do matching_improved.is_confident."""
    if not ranking:
        return False
    best_name, best_dist = ranking[0]
    if best_dist > threshold:
        return False
    if len(ranking) < 2:
        return True
    _, second_dist = ranking[1]
    return (second_dist - best_dist) >= min_margin


def motion_energy(frames):
    """
    frames: (N, 21, 3), N >= 2. Mede o quanto a mao se moveu no total
    (soma das distancias entre frames consecutivos, landmark a
    landmark, ponderada pelos mesmos pesos por landmark). Usado pra
    decidir se a mao esta "parada" (letra estatica) ou "em movimento"
    (letra dinamica) -- e o gatilho pra saber quando vale rodar a
    classificacao de sequencia.

    Valor de referencia: com a mao parada (tremor natural), fica perto
    de 0. Um gesto de letra com movimento (H, J, K, X, Z...) costuma
    passar bem de 1.0 num buffer de ~0.8s. AJUSTE o limiar no script
    que usa esta funcao observando os valores reais da sua webcam.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.shape[0] < 2:
        return 0.0
    diffs = frames[1:] - frames[:-1]                   # (N-1, 21, 3)
    dist_per_point = np.linalg.norm(diffs, axis=2)        # (N-1, 21)
    weighted = dist_per_point * LANDMARK_WEIGHTS.T          # (N-1, 21)
    per_frame = weighted.sum(axis=1) / LANDMARK_WEIGHTS.sum()
    return float(per_frame.sum())


# ---------------------------------------------------------------------
# Persistencia (mesmo padrao de acumulo do capture_signatures.py)
# ---------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Aviso: nao foi possivel ler {os.path.basename(path)} ({e}). Comecando do zero.")
        return {}


def as_sequence_sample_list(value):
    """
    Normaliza uma entrada do arquivo bruto (que pode ser uma unica
    sequencia (T,21,3) ou uma lista de sequencias (n,T,21,3)) para
    SEMPRE uma lista de sequencias, pra poder concatenar.
    """
    arr = np.array(value)
    if arr.ndim == 3:  # sequencia unica -> vira lista com 1 amostra
        return [arr.tolist()]
    return arr.tolist()


def save_movement_data(samples, output_path, raw_path, average=True):
    """
    samples: dict nome -> lista de sequencias NOVAS (cada uma ja
    reamostrada para um T fixo) capturadas nesta sessao.
    Acrescenta ao pool bruto (raw_path, nunca sobrescrito, so cresce)
    e recalcula o arquivo final a partir do pool completo -- mesma
    logica do save_data() do capture_signatures.py.
    Retorna dict {nome: total_no_pool} ou None se nao havia nada novo.
    """
    targets = {name: s for name, s in samples.items() if s}
    if not targets:
        return None

    raw_pool = load_json(raw_path)
    output_data = load_json(output_path)

    updated = []
    for name, new_seqs in targets.items():
        existing = as_sequence_sample_list(raw_pool[name]) if name in raw_pool else []
        combined = existing + new_seqs
        raw_pool[name] = combined

        if average:
            arr = np.array(combined)  # (n, T, 21, 3) -- exige mesmo T em todas as amostras
            mean_seq = np.mean(arr, axis=0)
            output_data[name] = np.round(mean_seq, 5).tolist()
        else:
            output_data[name] = combined

        updated.append(name)

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return {name: len(raw_pool[name]) for name in updated}


def load_movement_signatures(path):
    """Carrega movement.json -> dict nome -> ndarray (n_amostras, T, 21, 3)."""
    raw = load_json(path)
    signatures = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.ndim == 3:  # uma unica sequencia media (T, 21, 3)
            signatures[name] = arr[np.newaxis, :, :, :]
        else:
            signatures[name] = arr
    return signatures
