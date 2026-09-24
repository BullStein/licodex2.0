"""
Servidor de Streaming - LIBRAS (celular -> PC)
--------------------------------------------------
O CELULAR só captura a câmera e mostra o resultado. Quem faz o
trabalho pesado (MediaPipe + classificação) continua sendo o PC,
exatamente como no translator.py -- só que em vez de ler da webcam
local, os frames chegam pela rede (Wi-Fi local) via WebSocket.

Por que só 1 mão (sem "mão de comando" como no translator.py):
    No celular, uma mão geralmente está seguran do o aparelho, então
    só a mão livre fica disponível pra fazer o sinal. Os comandos
    (BACKSPACE/SPACE/CLEAR) viram botões na tela do próprio app
    Android, tratados 100% localmente no celular -- não precisam do
    servidor.

Protocolo (bem simples, por cima do WebSocket):
    - Cliente (celular) conecta em ws://<IP_DO_PC>:8765
    - Cliente manda mensagens BINÁRIAS: um frame JPEG por mensagem
    - Servidor responde com uma mensagem de TEXTO (JSON) por frame:
        {"letra": "A", "fonte": "estatica", "energia": 0.12}
        {"letra": "-", "fonte": null, "energia": 0.0}   (sem mão / sem reconhecimento confiante)
      "fonte" é "estatica" ou "movimento", indicando de onde veio o palpite.
    - Cliente pode mandar mensagens de TEXTO de controle:
        "reset"  -> limpa o buffer de movimento dessa conexão (útil
                    se o usuário trocou de gesto abruptamente)

Cada conexão (cada celular) tem seu PRÓPRIO estado (buffer de
suavização + buffer de movimento), então dá pra ter mais de um
celular conectado ao mesmo tempo sem um bagunçar o outro.

Pré-requisitos:
    pip install websockets opencv-python mediapipe numpy --break-system-packages
    - data.json, movement.json (opcionais: sem eles, só cai pra "sem
      reconhecimento" e o servidor avisa no console)
    - matching_improved.py, dynamic_signatures.py na mesma pasta

Como usar:
    python stream_server.py
    python stream_server.py --host 0.0.0.0 --port 8765
    python stream_server.py --max-hands 1

No Android, aponte o app pro IP local do PC que aparece impresso ao
iniciar (ex.: ws://192.168.0.42:8765).
"""

import os
import sys
import json
import math
import time
import socket
import asyncio
import argparse
from collections import deque, Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(BASE_DIR, "data.json")
MOVEMENT_DATA_PATH = os.path.join(BASE_DIR, "movement.json")

WRIST = 0
MIDDLE_MCP = 9

LETTER_CONFIDENCE_THRESHOLD = 0.55
LETTER_MIN_MARGIN = 0.04

MOTION_BUFFER_MAXLEN = 40
MOTION_MIN_FRAMES = 8
MOTION_ENERGY_THRESHOLD = 1.0
MOVEMENT_CONFIDENCE_THRESHOLD = 0.6
MOVEMENT_MIN_MARGIN = 0.04
KNN_K = 3
SMOOTH_MAXLEN = 8


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execução)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído:", MODEL_PATH)


def load_signatures(path, label):
    if not os.path.exists(path):
        print(f"Aviso: {os.path.basename(path)} não encontrado ({label} desativado até você gerá-lo).")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    signatures = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        signatures[name] = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    return signatures


def normalize_landmarks(landmarks):
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array([[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
                      for lm in landmarks])


def build_landmarker(max_hands=1):
    """
    Isolado numa função própria de propósito: os testes automatizados
    substituem essa função por uma versão falsa, pra validar o resto
    do servidor (protocolo, concorrência, erros) sem precisar do
    modelo .task de verdade nem de uma mão real na frente da câmera.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def decode_jpeg(jpeg_bytes):
    """Isolado numa função própria pelo mesmo motivo (testável sem cv2 real, se preciso)."""
    import cv2
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def classify_hand(normalized_pose, signatures, threshold, min_margin):
    if not signatures:
        return None
    from matching_improved import classify_ranked_weighted, is_confident
    ranking = classify_ranked_weighted(normalized_pose, signatures, k=KNN_K)
    if not is_confident(ranking, threshold, min_margin):
        return None
    return ranking[0][0]


def classify_movement(buffer, signatures):
    if not signatures or len(buffer) < MOTION_MIN_FRAMES:
        return None
    import dynamic_signatures as dyn
    query = np.array(buffer, dtype=np.float64)
    ranking = dyn.classify_sequence_ranked(query, signatures, k=KNN_K)
    if not dyn.is_confident(ranking, MOVEMENT_CONFIDENCE_THRESHOLD, MOVEMENT_MIN_MARGIN):
        return None
    return ranking[0][0]


def motion_energy(buffer):
    if len(buffer) < 2:
        return 0.0
    import dynamic_signatures as dyn
    return dyn.motion_energy(np.array(buffer))


class ClientSession:
    """Estado independente por conexão (por celular)."""

    def __init__(self, conn_id):
        self.conn_id = conn_id
        self.letter_buffer = deque(maxlen=SMOOTH_MAXLEN)
        self.motion_buffer = deque(maxlen=MOTION_BUFFER_MAXLEN)
        self.frame_count = 0
        self.start_time = time.time()

    def next_timestamp_ms(self):
        self.frame_count += 1
        return int((time.time() - self.start_time) * 1000)

    def reset_motion(self):
        self.motion_buffer.clear()

    def smooth(self, value):
        self.letter_buffer.append(value)
        most_common, _ = Counter(self.letter_buffer).most_common(1)[0]
        return most_common


def process_frame(session, landmarker, rgb_frame, letter_signatures, movement_signatures):
    """
    Núcleo puro (sem I/O de rede) que processa 1 frame já decodificado
    em RGB e devolve o dict de resposta. Separado do handler de
    WebSocket pra ficar fácil de testar isoladamente.
    """
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = session.next_timestamp_ms()
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        session.reset_motion()
        return {"letra": "-", "fonte": None, "energia": 0.0}

    hand_landmarks = result.hand_landmarks[0]
    normalized_pose = normalize_landmarks(hand_landmarks)
    session.motion_buffer.append(normalized_pose)

    raw_letter = classify_hand(normalized_pose, letter_signatures, LETTER_CONFIDENCE_THRESHOLD, LETTER_MIN_MARGIN)
    static_letter = session.smooth(raw_letter or "?")

    energy = motion_energy(session.motion_buffer)
    movement_letter = None
    if movement_signatures and energy > MOTION_ENERGY_THRESHOLD:
        movement_letter = classify_movement(session.motion_buffer, movement_signatures)

    if movement_letter is not None:
        return {"letra": movement_letter, "fonte": "movimento", "energia": round(float(energy), 3)}
    return {"letra": static_letter, "fonte": "estatica" if raw_letter else None, "energia": round(float(energy), 3)}


async def handle_connection(websocket, landmarker, letter_signatures, movement_signatures):
    conn_id = f"{websocket.remote_address}" if hasattr(websocket, "remote_address") else "cliente"
    session = ClientSession(conn_id)
    print(f"[+] Conectado: {conn_id}")
    try:
        async for message in websocket:
            if isinstance(message, str):
                if message.strip().lower() == "reset":
                    session.reset_motion()
                    await websocket.send(json.dumps({"ok": True, "msg": "buffer de movimento resetado"}))
                continue

            rgb_frame = decode_jpeg(message)
            if rgb_frame is None:
                await websocket.send(json.dumps({"erro": "frame inválido (não decodificou como JPEG)"}))
                continue

            try:
                response = process_frame(session, landmarker, rgb_frame, letter_signatures, movement_signatures)
            except Exception as e:  # nunca deixa uma exceção de 1 frame derrubar a conexão inteira
                response = {"erro": f"falha ao processar frame: {e}"}

            await websocket.send(json.dumps(response, ensure_ascii=False))
    except Exception as e:
        print(f"[!] Conexão {conn_id} encerrada com erro: {e}")
    finally:
        print(f"[-] Desconectado: {conn_id}")


def detect_lan_ip():
    """Truque clássico: 'conecta' via UDP num IP externo (nada é enviado de
    verdade) só pra descobrir qual interface de rede local o SO usaria."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


async def main_async(args):
    import websockets

    landmarker = build_landmarker(max_hands=1)
    letter_signatures = load_signatures(DATA_PATH, "letras")
    movement_signatures = load_signatures(MOVEMENT_DATA_PATH, "letras de movimento")
    if movement_signatures:
        letter_signatures = {n: v for n, v in letter_signatures.items() if n not in movement_signatures}

    async def handler(websocket):
        await handle_connection(websocket, landmarker, letter_signatures, movement_signatures)

    lan_ip = detect_lan_ip()
    print(f"Servidor no ar em ws://{lan_ip}:{args.port}  (bind: {args.host}:{args.port})")
    print("Aponte o app Android pra esse endereço (mesma rede Wi-Fi).")
    print("CTRL+C encerra.")

    async with websockets.serve(handler, args.host, args.port, max_size=5 * 1024 * 1024):
        await asyncio.Future()  # roda pra sempre


def main():
    parser = argparse.ArgumentParser(description="Servidor de streaming LIBRAS (celular -> PC).")
    parser.add_argument("--host", default="0.0.0.0", help="Interface pra escutar (padrão: 0.0.0.0, todas)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")


if __name__ == "__main__":
    main()
