"""
Captura de Assinaturas de Landmarks - LIBRAS
---------------------------------------------
Gera/atualiza SEIS arquivos:
    - data.json             -> assinaturas MEDIAS das LETRAS ESTATICAS (mao de LIBRAS)
    - data_raw.json         -> POOL de amostras cruas de letras estaticas
    - movement.json         -> assinaturas MEDIAS das LETRAS COM MOVIMENTO (ex.: H, J, K, X, Z)
    - movement_raw.json     -> POOL de amostras cruas de letras com movimento (sequencias)
    - commands.json         -> assinaturas MEDIAS/POOL dos COMANDOS (mao de comando)
    - commands_raw.json     -> POOL de todas as amostras cruas de comandos ja capturadas

ATENÇÃO -- ESTE ARQUIVO PRECISA DE dynamic_signatures.py (import dynamic_signatures as dyn),
QUE NUNCA FOI COLADO NA CONVERSA COM O CLAUDE. Cole o conteúdo real desse
arquivo aqui do lado antes de rodar, senão o "import dynamic_signatures as dyn"
abaixo vai falhar com ModuleNotFoundError.

A assinatura ESTATICA e a posicao dos 21 landmarks da mao (x, y, z),
NORMALIZADOS em relacao ao pulso e escalados pelo tamanho da mao.

A assinatura DE MOVIMENTO e uma SEQUENCIA de poses normalizadas ao
longo de uma janela de tempo curta (--record-seconds), reamostrada
para um numero fixo de frames -- ver dynamic_signatures.py pra
detalhes de como a comparacao funciona.

MODOS (tecla "8" alterna entre eles):
    - Modo LETRA (padrao): aperte a tecla da letra (A-Z, "9"=Ç) pra
      capturar uma amostra daquela letra.
        * Se a letra estiver na lista de movimento (--movement-letters,
          padrao: H,J,K,X,Z), o aperto da tecla NAO captura so 1 frame
          -- ele inicia uma GRAVACAO de --record-seconds segundos.
          Faca o movimento completo do gesto durante essa janela; ao
          final, a sequencia inteira vira UMA amostra de movimento.
        * Se nao estiver na lista, captura instantanea de 1 frame,
          como sempre.
    - Modo COMANDO: navega por uma LISTA de comandos pre-definidos
      (COMMANDS, mais abaixo) e captura amostra do que estiver
      selecionado:
          "4" = comando anterior da lista
          "6" = proximo comando da lista
          "5" ou ESPACO = capturar amostra do comando selecionado

Controles gerais (funcionam nos dois modos):
    - "8" = alternar entre modo LETRA e modo COMANDO
    - "1" = salvar a MEDIA (letras estaticas -> data.json | letras de
            movimento -> movement.json | comandos -> commands.json),
            calculada em cima de TODO o pool acumulado
    - "2" = salvar TODAS as amostras do pool acumulado, sem tirar media
    - "3" = resetar as amostras do alvo atual NESTA SESSAO (nao mexe
            no pool ja salvo em disco)
    - "0" (ou ESC) = sair sem salvar

IMPORTANTE - como funciona o acumulo entre execucoes:
    Toda vez que voce aperta "1" ou "2", as amostras capturadas nesta
    sessao sao ACRESCENTADAS ao pool bruto (data_raw.json /
    movement_raw.json / commands_raw.json), que nunca e sobrescrito
    -- so cresce. O arquivo final e sempre recalculado a partir do
    pool completo (tudo que voce ja capturou em qualquer execucao
    anterior + agora).

Como usar:
    python capture_signatures.py
    python capture_signatures.py --movement-letters H,J,K,X,Z,Ç
    python capture_signatures.py --record-seconds 1.3
    python capture_signatures.py --movement-letters "" --record-seconds 1.0
        (lista vazia = trata TODAS as letras como estaticas, desliga
        gravacao por tempo; util se voce quiser recapturar do zero)
"""

import os
import json
import math
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import dynamic_signatures as dyn

# ---------------------------------------------------------------------
# Modelo (reaproveita o mesmo hand_landmarker.task dos outros scripts)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(BASE_DIR, "data.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data_raw.json")
MOVEMENT_DATA_PATH = os.path.join(BASE_DIR, "movement.json")
RAW_MOVEMENT_DATA_PATH = os.path.join(BASE_DIR, "movement_raw.json")
COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands.json")
RAW_COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands_raw.json")

WINDOW_NAME = "Captura de Assinaturas - LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 620  # janela de tamanho medio, nao fullscreen

TARGET_FPS = 60          # teto de FPS
MIN_TARGET_FPS = 30      # piso aceitavel (informativo, pra referencia)
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

WRIST = 0
MIDDLE_MCP = 9  # usado como referencia de "tamanho da mao"

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["Ç"]

# Letras cujo gesto envolve MOVIMENTO por padrao. Sobrescrito por
# --movement-letters na linha de comando (ver main()).
DEFAULT_MOVEMENT_LETTERS = {"H", "J", "K", "X", "Z"}

# Duracao padrao da janela de gravacao (em segundos) pra cada amostra
# de letra de movimento. Sobrescrito por --record-seconds.
DEFAULT_RECORD_SECONDS = 1.0

# Numero de frames pro qual toda sequencia capturada e reamostrada
# antes de virar uma amostra (ver dynamic_signatures.resample_sequence).
SEQUENCE_LENGTH = dyn.DEFAULT_SEQUENCE_LENGTH

# Lista de comandos disponiveis para captura no modo COMANDO.
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # minimo
    (0, 17),                                  # base da palma
]


def parse_movement_letters(raw):
    """'H,J,K,X,Z' -> {'H','J','K','X','Z'}. String vazia -> conjunto vazio."""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execucao)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluido:", MODEL_PATH)


def normalize_landmarks(landmarks):
    """
    Retorna array (21, 3) [x, y, z], normalizados:
    - Origem deslocada para o pulso (landmark 0)
    - Escala dividida pela distancia pulso -> base do dedo medio
    """
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt(
        (ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2
    )
    scale = scale if scale > 1e-6 else 1e-6

    normalized = []
    for lm in landmarks:
        normalized.append([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return np.array(normalized, dtype=np.float64)


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Aviso: não foi possível ler {os.path.basename(path)} existente ({e}). "
              "Um novo arquivo será criado a partir do zero.")
        return {}


def as_sample_list(value):
    """
    Normaliza uma entrada de arquivo (que pode ser uma pose unica
    (21,3) ou uma lista de poses (n,21,3)) para SEMPRE uma lista de
    poses, pra poder concatenar.
    """
    arr = np.array(value)
    if arr.ndim == 2:  # pose unica -> vira lista com 1 amostra
        return [arr.tolist()]
    return arr.tolist()


def save_data(samples, output_path, raw_path, average=True):
    """
    Acrescenta as amostras capturadas nesta sessao ao POOL bruto
    (raw_path), que nunca e sobrescrito -- so cresce entre execucoes.
    Depois recalcula o arquivo final (output_path) a partir do pool
    completo (amostras antigas + desta sessao).
    """
    targets_with_samples = {name: s for name, s in samples.items() if s}
    if not targets_with_samples:
        print("Nenhuma amostra capturada ainda, nada foi salvo.")
        return

    raw_pool = load_json(raw_path)
    output_data = load_json(output_path)

    updated_targets = []
    for name, new_samples in targets_with_samples.items():
        existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
        combined_raw = existing_raw + new_samples
        raw_pool[name] = combined_raw

        if average:
            arr = np.array(combined_raw)
            mean_pose = np.mean(arr, axis=0)
            output_data[name] = np.round(mean_pose, 5).tolist()
        else:
            output_data[name] = combined_raw

        updated_targets.append(name)

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(output_path)} atualizado em: {output_path}")
    for name in sorted(updated_targets):
        print(f"  {name}: {len(raw_pool[name])} amostras no pool total (pool salvo em {os.path.basename(raw_path)})")
    print(f"  Total de alvos no arquivo final: {len(output_data)} -> {sorted(output_data.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Captura assinaturas de landmarks (letras/comandos/movimento) pra LIBRAS.")
    parser.add_argument("--movement-letters", default=",".join(sorted(DEFAULT_MOVEMENT_LETTERS)),
                         help="Letras tratadas como GESTO DE MOVIMENTO, separadas por vírgula "
                              "(padrão: H,J,K,X,Z). Use \"\" pra desligar e tratar tudo como estático.")
    parser.add_argument("--record-seconds", type=float, default=DEFAULT_RECORD_SECONDS,
                         help=f"Duração da janela de gravação de um gesto de movimento (padrão: {DEFAULT_RECORD_SECONDS}s)")
    args = parser.parse_args()

    movement_letters = parse_movement_letters(args.movement_letters)
    record_seconds = args.record_seconds

    unknown = movement_letters - set(LETTERS)
    if unknown:
        print(f"Aviso: {sorted(unknown)} em --movement-letters não são letras válidas (A-Z, Ç) e serão ignoradas.")
        movement_letters -= unknown

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,  # captura de proposito so com 1 mao por vez (mais limpo)
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # --- Estado de modo ---
    mode = "letter"  # ou "command"
    current_letter = None
    current_command_index = 0

    # amostras acumuladas nesta sessao (so o que foi capturado agora;
    # o pool completo entre sessoes vive em data_raw.json / movement_raw.json / commands_raw.json)
    samples_letters = defaultdict(list)      # letras ESTATICAS: lista de poses (21,3)
    samples_movements = defaultdict(list)    # letras de MOVIMENTO: lista de sequencias (T,21,3)
    samples_commands = defaultdict(list)

    # --- Estado da gravacao de movimento em andamento (None = parado) ---
    recording_letter = None
    recording_buffer = []
    recording_start = 0.0

    frame_timestamp_ms = 0
    start_time = time.time()
    last_landmarks = None

    print("Captura iniciada. Modo atual: LETRA")
    if movement_letters:
        print(f"Letras de movimento ({', '.join(sorted(movement_letters))}): tecle a letra e faca o "
              f"gesto durante {record_seconds:.1f}s.")
    else:
        print("Nenhuma letra configurada como movimento (--movement-letters vazio) -- tudo captura instantâneo.")
    print("Letra estatica: tecle A-Z pra capturar 1 frame | Comando: '4'/'6' navega, '5'/ESPACO captura")
    print("'8' = alternar modo | '1' = salvar médias | '2' = salvar tudo | '3' = resetar alvo atual (sessão) | '0'/ESC = sair")

    while cap.isOpened():
        loop_start = time.time()

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        last_landmarks = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
            last_landmarks = hand_landmarks

        # --- Se uma gravacao de movimento esta em andamento, acumula frames ---
        is_recording = recording_letter is not None
        if is_recording:
            if last_landmarks is not None:
                recording_buffer.append(normalize_landmarks(last_landmarks))

            elapsed_recording = time.time() - recording_start
            if elapsed_recording >= record_seconds:
                if len(recording_buffer) >= 2:
                    resampled = dyn.resample_sequence(recording_buffer, SEQUENCE_LENGTH)
                    samples_movements[recording_letter].append(resampled.tolist())
                    print(f"Amostra de MOVIMENTO capturada para '{recording_letter}' "
                          f"({len(recording_buffer)} frames -> reamostrado p/ {SEQUENCE_LENGTH}); "
                          f"total nesta sessão: {len(samples_movements[recording_letter])}")
                else:
                    print(f"Gravação de '{recording_letter}' cancelada: mão não detectada durante a janela.")
                recording_letter = None
                recording_buffer = []
                is_recording = False

        # --- HUD ---
        h, w, _ = frame.shape
        draw_transparent_box(frame, (0, 0), (460, 90), color=(0, 0, 0), alpha=0.55)

        if mode == "letter":
            if is_recording:
                remaining = max(0.0, record_seconds - (time.time() - recording_start))
                info = f"[GRAVANDO MOVIMENTO] '{recording_letter}' | faltam {remaining:.1f}s | frames: {len(recording_buffer)}"
                controls = "Faça o gesto completo até o tempo acabar..."
            else:
                n_samples = len(samples_letters[current_letter]) if current_letter else 0
                n_mov = len(samples_movements[current_letter]) if current_letter in movement_letters else 0
                tag = " (MOVIMENTO)" if current_letter in movement_letters else ""
                count = n_mov if current_letter in movement_letters else n_samples
                info = f"[MODO LETRA] Letra atual: {current_letter or '-'}{tag} | Amostras (sessão): {count}"
                controls = "Letra=captura | 8=modo comando | 1/2=salvar | 3=resetar | 0=sair"
        else:
            cmd_name = COMMANDS[current_command_index]
            n_samples = len(samples_commands[cmd_name])
            info = f"[MODO COMANDO] Comando ({current_command_index + 1}/{len(COMMANDS)}): {cmd_name} | Amostras (sessao): {n_samples}"
            controls = "4/6=navega | 5/ESPACO=captura | 8=modo letra | 1/2=salvar | 3=resetar | 0=sair"

        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, controls, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Painel com a lista completa de comandos (modo comando) ---
        if mode == "command":
            list_top = 100
            list_height = 30 + len(COMMANDS) * 26
            draw_transparent_box(frame, (0, list_top), (260, list_top + list_height), color=(0, 0, 0), alpha=0.55)
            cv2.putText(frame, "Comandos disponiveis:", (10, list_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            for i, cmd_option in enumerate(COMMANDS):
                is_selected = i == current_command_index
                color = (0, 255, 0) if is_selected else (200, 200, 200)
                prefix = "> " if is_selected else "   "
                n = len(samples_commands[cmd_option])
                cv2.putText(frame, f"{prefix}{cmd_option} ({n})", (10, list_top + 48 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_selected else 1, cv2.LINE_AA)

        # --- Painel com as letras de movimento configuradas (modo letra) ---
        if mode == "letter" and movement_letters and not is_recording:
            mov_names = sorted(movement_letters)
            mov_box_h = 30 + len(mov_names) * 20
            draw_transparent_box(frame, (w - 180, 0), (w, mov_box_h), color=(0, 0, 0), alpha=0.5)
            cv2.putText(frame, "Config. p/ movimento:", (w - 170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            for i, name in enumerate(mov_names):
                n = len(samples_movements[name])
                cv2.putText(frame, f"- {name} ({n})", (w - 170, 40 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Barra de progresso da gravacao de movimento ---
        if is_recording:
            fraction = min(1.0, (time.time() - recording_start) / record_seconds)
            bar_top_left = (10, 100)
            bar_bottom_right = (410, 120)
            cv2.rectangle(frame, bar_top_left, bar_bottom_right, (100, 100, 100), 1)
            filled_w = int((bar_bottom_right[0] - bar_top_left[0]) * fraction)
            cv2.rectangle(frame, bar_top_left, (bar_top_left[0] + filled_w, bar_bottom_right[1]), (0, 200, 255), -1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF

        if is_recording:
            # Enquanto grava, ignora todas as outras teclas (exceto sair) pra nao
            # bagunçar o estado no meio da captura do gesto.
            if key == ord("0") or key == 27:
                break
            # --- Limitador de FPS ---
            elapsed = time.time() - loop_start
            if elapsed < MIN_FRAME_INTERVAL:
                time.sleep(MIN_FRAME_INTERVAL - elapsed)
            continue

        if key == ord("0") or key == 27:  # "0" ou ESC
            break

        elif key == ord("8"):
            mode = "command" if mode == "letter" else "letter"
            print(f"Modo alternado para: {'COMANDO' if mode == 'command' else 'LETRA'}")

        elif key == ord("1"):
            if mode == "letter":
                save_data(samples_letters, DATA_PATH, RAW_DATA_PATH, average=True)
                result_mov = dyn.save_movement_data(samples_movements, MOVEMENT_DATA_PATH, RAW_MOVEMENT_DATA_PATH, average=True)
                if result_mov:
                    print(f"{os.path.basename(MOVEMENT_DATA_PATH)} atualizado: "
                          + ", ".join(f"{n}: {c} amostras no pool" for n, c in sorted(result_mov.items())))
            else:
                save_data(samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH, average=True)

        elif key == ord("2"):
            if mode == "letter":
                save_data(samples_letters, DATA_PATH, RAW_DATA_PATH, average=False)
                result_mov = dyn.save_movement_data(samples_movements, MOVEMENT_DATA_PATH, RAW_MOVEMENT_DATA_PATH, average=False)
                if result_mov:
                    print(f"{os.path.basename(MOVEMENT_DATA_PATH)} atualizado (pool completo, sem média): "
                          + ", ".join(f"{n}: {c} amostras" for n, c in sorted(result_mov.items())))
            else:
                save_data(samples_commands, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH, average=False)

        elif key == ord("3"):
            if mode == "letter" and current_letter:
                samples_letters[current_letter] = []
                samples_movements[current_letter] = []
                print(f"Amostras da letra {current_letter} resetadas (só as desta sessão).")
            elif mode == "command":
                cmd_name = COMMANDS[current_command_index]
                samples_commands[cmd_name] = []
                print(f"Amostras do comando {cmd_name} resetadas (só as desta sessão).")

        elif mode == "command" and key == ord("4"):
            current_command_index = (current_command_index - 1) % len(COMMANDS)

        elif mode == "command" and key == ord("6"):
            current_command_index = (current_command_index + 1) % len(COMMANDS)

        elif mode == "command" and key in (ord("5"), 32):  # "5" ou barra de espaço
            cmd_name = COMMANDS[current_command_index]
            if last_landmarks is not None:
                normalized = normalize_landmarks(last_landmarks)
                samples_commands[cmd_name].append(normalized.tolist())
                print(f"Amostra capturada para comando '{cmd_name}' (total nesta sessão: {len(samples_commands[cmd_name])})")
            else:
                print(f"Comando '{cmd_name}' selecionado, mas nenhuma mão detectada no frame.")

        elif mode == "letter":
            ch = "Ç" if key == ord("9") else (chr(key).upper() if 32 <= key <= 126 else "")
            if ch in LETTERS:
                current_letter = ch
                if ch in movement_letters:
                    # Inicia a gravacao por tempo em vez de capturar so 1 frame.
                    recording_letter = ch
                    recording_buffer = []
                    recording_start = time.time()
                    print(f"Gravando gesto de movimento para '{ch}' por {record_seconds:.1f}s...")
                elif last_landmarks is not None:
                    normalized = normalize_landmarks(last_landmarks)
                    samples_letters[ch].append(normalized.tolist())
                    print(f"Amostra capturada para '{ch}' (total nesta sessão: {len(samples_letters[ch])})")
                else:
                    print(f"Letra '{ch}' selecionada, mas nenhuma mão detectada no frame.")

        # --- Limitador de FPS: garante um ritmo estavel (nao passa de TARGET_FPS) ---
        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
