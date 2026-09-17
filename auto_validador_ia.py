"""
Auto-Validador com Juiz de IA (Ollama) - LIBRAS
--------------------------------------------------
ATENÇÃO -- ESTE ARQUIVO PRECISA DE matching_improved.py E
dynamic_signatures.py, QUE NUNCA FORAM COLADOS NA CONVERSA COM O
CLAUDE. Cole o conteúdo real desses dois arquivos aqui do lado antes
de rodar. Prefira usar auto_validador_nn.py (rede neural) enquanto
isso -- ele só depende de dynamic_signatures.py (não de
matching_improved.py nem do Ollama).

Versão com suporte a TRÊS tipos de alvo:
    - Letras ESTÁTICAS (uma pose só)          -> data.json / data_raw.json
    - Comandos (BACKSPACE/SPACE/CLEAR)        -> commands.json / commands_raw.json
    - Letras de MOVIMENTO (H, J, K, X, Z...)  -> movement.json / movement_raw.json

Em vez de você apertar S/N toda vez, quem julga o palpite primeiro é
uma IA local rodando no Ollama (ver ollama_config.py pra trocar de
modelo entre casa/notebook). Se a IA ficar em dúvida ou o Ollama cair,
o programa devolve o controle pra você (fluxo manual S/N), nunca
treina a base "no escuro".

DOIS JEITOS DE CAPTURAR:
    1) ESPAÇO = "foto" instantânea -> classifica letra estática (mão
       esquerda) + comando (mão direita), igual ao validador.py normal.
    2) "R"    = inicia uma GRAVAÇÃO de --record-seconds segundos (padrão
       1.0s) da mão esquerda -> ao final, a sequência inteira é
       comparada contra movement.json (dynamic_signatures.py, mesma
       lógica usada no capture_signatures.py/translator.py). Use isso
       pra H, J, K, X, Z (ou a lista que você configurar em
       --movement-letters).

Fluxo de decisão (pros três tipos):
    - IA diz "correto" com confiança >= --auto-confirm  -> salva sozinho.
    - IA diz "errado" com confiança >= --auto-correct E a sugestão dela
      já existe na base -> corrige sozinho pra classe sugerida.
    - Qualquer outro caso (incerto, baixa confiança, Ollama offline)
      -> cai pro modo MANUAL (S/N, com correção manual em seguida).

Pré-requisitos:
    - data.json / commands.json / movement.json (gerados pelo
      capture_signatures.py e/ou batch_train_from_images.py)
    - matching_improved.py, dynamic_signatures.py, ollama_config.py e
      ai_judge.py na mesma pasta
    - Ollama rodando localmente com o modelo do perfil ativo já baixado

Como usar:
    python auto_validador_ia.py
    python auto_validador_ia.py --profile notebook
    python auto_validador_ia.py --movement-letters H,J,K,X,Z,Ç
    python auto_validador_ia.py --record-seconds 1.2
    python auto_validador_ia.py --disable-ai      (vira o validador manual puro)

Controles:
    - ESPAÇO = foto instantânea (letra estática + comando)
    - "R"    = grava um gesto de MOVIMENTO (janela de tempo)
    - "S"/"N" = só aparecem no modo MANUAL (IA incerta/offline)
    - "C"    = cancela uma gravação de movimento em andamento
    - "0" ou ESC = sair
"""

import os
import json
import math
import time
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from matching_improved import classify_ranked_weighted
import dynamic_signatures as dyn
from ai_judge import judge_prediction
from ollama_config import get_active_settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_PATH = os.path.join(BASE_DIR, "data.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data_raw.json")
COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands.json")
RAW_COMMANDS_DATA_PATH = os.path.join(BASE_DIR, "commands_raw.json")
MOVEMENT_DATA_PATH = os.path.join(BASE_DIR, "movement.json")
RAW_MOVEMENT_DATA_PATH = os.path.join(BASE_DIR, "movement_raw.json")

WINDOW_NAME = "Auto-Validador (IA) - LIBRAS"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 660

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

LETTER_HAND_LABEL = "Left"
COMMAND_HAND_LABEL = "Right"

WRIST = 0
MIDDLE_MCP = 9

LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)} | {"Ç"}
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]
DEFAULT_MOVEMENT_LETTERS = ["H", "J", "K", "X", "Z"]
KNN_K = 3

# --- Estados ---
STATE_IDLE = "idle"
STATE_RECORDING_MOVEMENT = "recording_movement"
STATE_MANUAL_CONFIRM_STATIC = "manual_confirm_static"      # letra estática e/ou comando
STATE_CORRECT_LETTER = "correct_letter"
STATE_CORRECT_COMMAND = "correct_command"
STATE_MANUAL_CONFIRM_MOVEMENT = "manual_confirm_movement"
STATE_CORRECT_MOVEMENT = "correct_movement"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo hand_landmarker.task (primeira execução)...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download concluído:", MODEL_PATH)


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def load_signatures(path, label):
    if not os.path.exists(path):
        print(f"Aviso: {os.path.basename(path)} não encontrado ({label} ficará sem reconhecer nada).")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    signatures = {}
    for name, value in raw.items():
        arr = np.array(value, dtype=np.float64)
        signatures[name] = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    return signatures


def as_sample_list(value):
    arr = np.array(value)
    return [arr.tolist()] if arr.ndim == 2 else arr.tolist()


def normalize_landmarks(landmarks):
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array([[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
                      for lm in landmarks])


def classify_ranked(normalized_pose, signatures):
    return classify_ranked_weighted(normalized_pose, signatures, k=KNN_K)


def draw_landmarks(frame, landmarks, color=(0, 200, 0)):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def confirm_static_sample(name, normalized_pose, output_path, raw_path):
    """Letras estáticas e comandos: uma pose só por amostra."""
    raw_pool = load_json(raw_path)
    output_data = load_json(output_path)
    existing_raw = as_sample_list(raw_pool[name]) if name in raw_pool else []
    combined_raw = existing_raw + [np.round(normalized_pose, 5).tolist()]
    raw_pool[name] = combined_raw
    output_data[name] = np.round(np.mean(np.array(combined_raw), axis=0), 5).tolist()
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pool, f, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return len(combined_raw)


def confirm_movement_sample(name, resampled_sequence):
    """Letras de movimento: uma SEQUÊNCIA (já reamostrada) por amostra.
    Reaproveita dynamic_signatures.save_movement_data, a mesma função
    usada pelo capture_signatures.py, pra manter o formato idêntico."""
    samples = {name: [resampled_sequence.tolist()]}
    result = dyn.save_movement_data(samples, MOVEMENT_DATA_PATH, RAW_MOVEMENT_DATA_PATH, average=True)
    return result.get(name) if result else None


def main():
    parser = argparse.ArgumentParser(description="Auto-validador com juiz de IA local (Ollama) pra LIBRAS, com suporte a letras de movimento.")
    parser.add_argument("--profile", default=None, help="Perfil do ollama_config.py a usar (ex.: casa, notebook)")
    parser.add_argument("--auto-confirm", type=float, default=0.75,
                         help="Confiança mínima da IA pra auto-confirmar um acerto (padrão: 0.75)")
    parser.add_argument("--auto-correct", type=float, default=0.8,
                         help="Confiança mínima da IA pra auto-corrigir pra outra classe sugerida (padrão: 0.8)")
    parser.add_argument("--disable-ai", action="store_true",
                         help="Desliga a IA e vira o fluxo 100%% manual (S/N)")
    parser.add_argument("--movement-letters", default=",".join(DEFAULT_MOVEMENT_LETTERS),
                         help="Letras tratadas como GESTO DE MOVIMENTO, separadas por vírgula (padrão: H,J,K,X,Z)")
    parser.add_argument("--record-seconds", type=float, default=1.0,
                         help="Duração da janela de gravação de um gesto de movimento (padrão: 1.0s)")
    args = parser.parse_args()

    movement_letters = {s.strip().upper() for s in args.movement_letters.split(",") if s.strip()}
    record_seconds = args.record_seconds
    sequence_length = dyn.DEFAULT_SEQUENCE_LENGTH

    if not args.disable_ai:
        model, host, profile_name = get_active_settings(args.profile)
        print(f"Juiz de IA ativo -> perfil '{profile_name}' | modelo '{model}' | host {host}")
    else:
        print("Juiz de IA desligado (--disable-ai). Rodando 100% manual.")
    print(f"Letras de MOVIMENTO configuradas: {sorted(movement_letters)} (janela: {record_seconds:.1f}s)")

    ensure_model()
    letter_signatures = load_signatures(DATA_PATH, "letras")
    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
    movement_signatures = dyn.load_movement_signatures(MOVEMENT_DATA_PATH)
    if movement_signatures:
        print(f"movement.json: {len(movement_signatures)} letra(s) já treinada(s) -> {sorted(movement_signatures.keys())}")
        # Mesma regra do translator.py: letras já confirmadas como dinâmicas saem da
        # classificação estática, pra não confundir/competir com a versão certa.
        letter_signatures = {name: v for name, v in letter_signatures.items() if name not in movement_signatures}

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
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

    frame_timestamp_ms = 0
    start_time = time.time()
    state = STATE_IDLE

    # --- ronda "estática" (SPACE): letra + comando ---
    letter_ranking, command_ranking = [], []
    letter_pose, command_pose = None, None
    letter_veredito = None

    # --- ronda "movimento" (R) ---
    recording_buffer = []
    recording_start = 0.0
    movement_ranking = []
    movement_sequence = None   # já reamostrada, pronta pra salvar
    movement_veredito = None
    correct_command_index = 0

    stats = {"auto_confirmadas": 0, "auto_corrigidas": 0, "manual_corretas": 0,
             "manual_erradas": 0, "manual_corrigidas": 0}
    last_feedback = ""
    last_feedback_until = 0

    print("Pronto. ESPAÇO = foto estática | R = gravar gesto de movimento | 0/ESC = sair")

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

        live_letter_landmarks, live_command_landmarks = None, None
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                if label == LETTER_HAND_LABEL:
                    live_letter_landmarks = hand_landmarks
                    draw_landmarks(frame, hand_landmarks, color=(0, 200, 0))
                elif label == COMMAND_HAND_LABEL:
                    live_command_landmarks = hand_landmarks
                    draw_landmarks(frame, hand_landmarks, color=(0, 140, 255))

        # --- Acumula frames durante uma gravação de movimento em andamento ---
        if state == STATE_RECORDING_MOVEMENT:
            if live_letter_landmarks is not None:
                recording_buffer.append(normalize_landmarks(live_letter_landmarks))

            elapsed_rec = time.time() - recording_start
            if elapsed_rec >= record_seconds:
                if len(recording_buffer) >= 2:
                    query = np.array(recording_buffer, dtype=np.float64)
                    movement_ranking = dyn.classify_sequence_ranked(query, movement_signatures, k=KNN_K) if movement_signatures else []
                    movement_sequence = dyn.resample_sequence(recording_buffer, sequence_length)

                    if not movement_ranking:
                        last_feedback = "Gravado, mas movement.json está vazio (nada pra comparar). Treine ao menos 1 amostra manual antes."
                        last_feedback_until = time.time() + 3.0
                        print(last_feedback)
                        state = STATE_IDLE
                    else:
                        print(f"\nGesto gravado ({len(recording_buffer)} frames). Ranking: {movement_ranking[:3]}")
                        needs_manual = True
                        if not args.disable_ai:
                            n_frames_txt = f"Sequência com {len(recording_buffer)} frames capturados."
                            movement_veredito = judge_prediction(movement_ranking, kind="movimento", extra_context=n_frames_txt)
                            print(f"[IA movimento] {movement_veredito}")
                            v, c, s = movement_veredito["veredito"], movement_veredito["confianca"], movement_veredito["sugestao"]
                            if v == "correto" and c >= args.auto_confirm:
                                n = confirm_movement_sample(movement_ranking[0][0], movement_sequence)
                                movement_signatures = dyn.load_movement_signatures(MOVEMENT_DATA_PATH)
                                stats["auto_confirmadas"] += 1
                                last_feedback = f"[AUTO] Movimento '{movement_ranking[0][0]}' confirmado pela IA ({n} amostras, conf={c:.2f})"
                                last_feedback_until = time.time() + 3.0
                                print(last_feedback)
                                needs_manual = False
                            elif v == "errado" and s in movement_signatures and c >= args.auto_correct:
                                n = confirm_movement_sample(s, movement_sequence)
                                movement_signatures = dyn.load_movement_signatures(MOVEMENT_DATA_PATH)
                                stats["auto_corrigidas"] += 1
                                last_feedback = f"[AUTO] Movimento corrigido pela IA para '{s}' ({n} amostras, conf={c:.2f})"
                                last_feedback_until = time.time() + 3.0
                                print(last_feedback)
                                needs_manual = False
                        state = STATE_MANUAL_CONFIRM_MOVEMENT if needs_manual else STATE_IDLE
                        if not needs_manual:
                            movement_ranking = []
                            movement_sequence = None
                else:
                    last_feedback = "Gravação cancelada: mão esquerda não detectada durante a janela."
                    last_feedback_until = time.time() + 2.5
                    print(last_feedback)
                    state = STATE_IDLE
                recording_buffer = []

        h, w, _ = frame.shape
        draw_box(frame, (0, 0), (380, 55))
        estado_txt = {
            STATE_IDLE: "Pronto (ESPAÇO=foto | R=grava movimento)",
            STATE_RECORDING_MOVEMENT: "GRAVANDO GESTO...",
            STATE_MANUAL_CONFIRM_STATIC: "Aguardando S/N (IA incerta - estático)",
            STATE_CORRECT_LETTER: "Corrigindo letra",
            STATE_CORRECT_COMMAND: "Corrigindo comando",
            STATE_MANUAL_CONFIRM_MOVEMENT: "Aguardando S/N (IA incerta - movimento)",
            STATE_CORRECT_MOVEMENT: "Corrigindo letra de movimento",
        }[state]
        cv2.putText(frame, f"Estado: {estado_txt}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        total_auto = stats["auto_confirmadas"] + stats["auto_corrigidas"]
        cv2.putText(frame, f"Auto: {total_auto} (confirm={stats['auto_confirmadas']} corrig={stats['auto_corrigidas']})  Manual: {stats['manual_corretas']}/{stats['manual_erradas']}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if state == STATE_RECORDING_MOVEMENT:
            remaining = max(0.0, record_seconds - (time.time() - recording_start))
            draw_box(frame, (0, 65), (410, 120), color=(0, 0, 0), alpha=0.55)
            cv2.putText(frame, f"Faça o gesto! Faltam {remaining:.1f}s | frames: {len(recording_buffer)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            fraction = min(1.0, (time.time() - recording_start) / record_seconds)
            cv2.rectangle(frame, (10, 100), (400, 112), (100, 100, 100), 1)
            cv2.rectangle(frame, (10, 100), (10 + int(390 * fraction), 112), (0, 200, 255), -1)
            cv2.putText(frame, "'C' cancela", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        elif state == STATE_MANUAL_CONFIRM_STATIC:
            fl = letter_ranking[0][0] if letter_ranking else None
            fc = command_ranking[0][0] if command_ranking else None
            draw_box(frame, (0, 65), (340, 220), color=(0, 0, 0), alpha=0.65)
            cv2.putText(frame, "IA em duvida -- confirme voce:", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Letra: {fl or '-'}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Comando: {fc or '-'}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            if letter_veredito:
                cv2.putText(frame, f"IA (letra): {letter_veredito['veredito']} ({letter_veredito['confianca']:.2f})",
                            (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "S=correto  N=errado (corrigir)", (10, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        elif state == STATE_CORRECT_LETTER:
            draw_box(frame, (0, 65), (400, 160), color=(0, 0, 40), alpha=0.7)
            cv2.putText(frame, "Qual era a letra certa?", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tecle A-Z / '9'=Ç  |  ESPACO = pular", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        elif state == STATE_CORRECT_COMMAND:
            list_top = 65
            list_height = 55 + len(COMMANDS) * 26
            draw_box(frame, (0, list_top), (320, list_top + list_height), color=(0, 0, 40), alpha=0.7)
            cv2.putText(frame, "Qual era o comando certo?", (10, list_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            for i, cmd_option in enumerate(COMMANDS):
                sel = i == correct_command_index
                color = (0, 255, 0) if sel else (200, 200, 200)
                cv2.putText(frame, f"{'> ' if sel else '   '}{cmd_option}", (10, list_top + 50 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if sel else 1, cv2.LINE_AA)
            cv2.putText(frame, "4/6=navega 5/ENTER=confirma ESPACO=pular", (10, list_top + list_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        elif state == STATE_MANUAL_CONFIRM_MOVEMENT:
            fm = movement_ranking[0][0] if movement_ranking else None
            draw_box(frame, (0, 65), (340, 190), color=(0, 0, 0), alpha=0.65)
            cv2.putText(frame, "IA em duvida (movimento) -- confirme:", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Gesto: {fm or '-'}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if movement_veredito:
                cv2.putText(frame, f"IA: {movement_veredito['veredito']} ({movement_veredito['confianca']:.2f})",
                            (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "S=correto  N=errado (corrigir)", (10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        elif state == STATE_CORRECT_MOVEMENT:
            opts = sorted(movement_letters)
            box_h = 60 + len(opts) * 22
            draw_box(frame, (0, 65), (300, 65 + box_h), color=(0, 0, 40), alpha=0.7)
            cv2.putText(frame, "Qual gesto de movimento era?", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tecle a letra certa:", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, ", ".join(opts), (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "ESPACO = pular", (10, 65 + box_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if last_feedback and time.time() < last_feedback_until:
            draw_box(frame, (0, h - 30), (w, h), color=(0, 90, 0), alpha=0.55)
            cv2.putText(frame, last_feedback, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(5) & 0xFF

        def set_feedback(msg, secs=2.5):
            nonlocal last_feedback, last_feedback_until
            last_feedback = msg
            last_feedback_until = time.time() + secs
            print(msg)

        def reset_static():
            nonlocal state, letter_ranking, command_ranking, letter_pose, command_pose, letter_veredito
            state = STATE_IDLE
            letter_ranking, command_ranking = [], []
            letter_pose, command_pose = None, None
            letter_veredito = None

        def reset_movement():
            nonlocal state, movement_ranking, movement_sequence, movement_veredito
            state = STATE_IDLE
            movement_ranking = []
            movement_sequence = None
            movement_veredito = None

        if key == ord("0") or key == 27:
            break

        elif key == ord("c") and state == STATE_RECORDING_MOVEMENT:
            state = STATE_IDLE
            recording_buffer = []
            set_feedback("Gravação cancelada pelo usuário.")

        elif key == ord("r") and state == STATE_IDLE:
            recording_buffer = []
            recording_start = time.time()
            state = STATE_RECORDING_MOVEMENT
            print(f"Gravando gesto de movimento por {record_seconds:.1f}s...")

        elif key == 32 and state == STATE_IDLE:
            letter_ranking = classify_ranked(normalize_landmarks(live_letter_landmarks), letter_signatures) if live_letter_landmarks is not None else []
            command_ranking = classify_ranked(normalize_landmarks(live_command_landmarks), command_signatures) if live_command_landmarks is not None else []
            letter_pose = normalize_landmarks(live_letter_landmarks) if live_letter_landmarks is not None else None
            command_pose = normalize_landmarks(live_command_landmarks) if live_command_landmarks is not None else None

            if not letter_ranking and not command_ranking:
                set_feedback("Foto tirada, mas nenhuma mão detectada. Tente de novo.")
                continue

            print(f"\nFoto tirada. Ranking letra: {letter_ranking[:3]} | comando: {command_ranking[:3]}")
            needs_manual = False

            if letter_ranking and not args.disable_ai:
                letter_veredito = judge_prediction(letter_ranking, kind="letra")
                print(f"[IA letra] {letter_veredito}")
                v, c, s = letter_veredito["veredito"], letter_veredito["confianca"], letter_veredito["sugestao"]
                if v == "correto" and c >= args.auto_confirm:
                    n = confirm_static_sample(letter_ranking[0][0], letter_pose, DATA_PATH, RAW_DATA_PATH)
                    letter_signatures = load_signatures(DATA_PATH, "letras")
                    letter_signatures = {name: vv for name, vv in letter_signatures.items() if name not in movement_signatures}
                    stats["auto_confirmadas"] += 1
                    set_feedback(f"[AUTO] Letra '{letter_ranking[0][0]}' confirmada pela IA ({n} amostras, conf={c:.2f})")
                elif v == "errado" and s in letter_signatures and c >= args.auto_correct:
                    n = confirm_static_sample(s, letter_pose, DATA_PATH, RAW_DATA_PATH)
                    letter_signatures = load_signatures(DATA_PATH, "letras")
                    letter_signatures = {name: vv for name, vv in letter_signatures.items() if name not in movement_signatures}
                    stats["auto_corrigidas"] += 1
                    set_feedback(f"[AUTO] Letra corrigida pela IA para '{s}' ({n} amostras, conf={c:.2f})")
                else:
                    needs_manual = True
            elif letter_ranking:
                needs_manual = True

            if command_ranking and not args.disable_ai:
                command_veredito = judge_prediction(command_ranking, kind="comando")
                print(f"[IA comando] {command_veredito}")
                v, c, s = command_veredito["veredito"], command_veredito["confianca"], command_veredito["sugestao"]
                if v == "correto" and c >= args.auto_confirm:
                    n = confirm_static_sample(command_ranking[0][0], command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                    stats["auto_confirmadas"] += 1
                    set_feedback(f"[AUTO] Comando '{command_ranking[0][0]}' confirmado pela IA ({n} amostras, conf={c:.2f})")
                elif v == "errado" and s in command_signatures and c >= args.auto_correct:
                    n = confirm_static_sample(s, command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                    command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                    stats["auto_corrigidas"] += 1
                    set_feedback(f"[AUTO] Comando corrigido pela IA para '{s}' ({n} amostras, conf={c:.2f})")
                else:
                    needs_manual = True
            elif command_ranking:
                needs_manual = True

            state = STATE_MANUAL_CONFIRM_STATIC if needs_manual else STATE_IDLE
            if not needs_manual:
                letter_ranking, command_ranking = [], []
                letter_pose, command_pose = None, None

        elif key == ord("s") and state == STATE_MANUAL_CONFIRM_STATIC:
            fed = []
            if letter_ranking and letter_pose is not None:
                name = letter_ranking[0][0]
                n = confirm_static_sample(name, letter_pose, DATA_PATH, RAW_DATA_PATH)
                letter_signatures = load_signatures(DATA_PATH, "letras")
                letter_signatures = {nm: vv for nm, vv in letter_signatures.items() if nm not in movement_signatures}
                stats["manual_corretas"] += 1
                fed.append(f"'{name}' ({n} amostras)")
            if command_ranking and command_pose is not None:
                name = command_ranking[0][0]
                n = confirm_static_sample(name, command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                stats["manual_corretas"] += 1
                fed.append(f"'{name}' ({n} amostras)")
            set_feedback("[OK manual] " + (" | ".join(fed) if fed else "nada para confirmar"))
            reset_static()

        elif key == ord("n") and state == STATE_MANUAL_CONFIRM_STATIC:
            if letter_ranking:
                stats["manual_erradas"] += 1
            if command_ranking:
                stats["manual_erradas"] += 1
            if letter_pose is not None:
                state = STATE_CORRECT_LETTER
            elif command_pose is not None:
                correct_command_index = 0
                state = STATE_CORRECT_COMMAND
            else:
                reset_static()

        elif state == STATE_CORRECT_LETTER:
            corrected = None
            if key == 32:
                pass
            elif key == ord("9"):
                corrected = "Ç"
            elif 32 <= key <= 126:
                ch = chr(key).upper()
                if ch in LETTERS:
                    corrected = ch
            if corrected is not None:
                n = confirm_static_sample(corrected, letter_pose, DATA_PATH, RAW_DATA_PATH)
                letter_signatures = load_signatures(DATA_PATH, "letras")
                letter_signatures = {nm: vv for nm, vv in letter_signatures.items() if nm not in movement_signatures}
                stats["manual_corrigidas"] += 1
                set_feedback(f"[Corrigido manual] letra -> '{corrected}' ({n} amostras)")
            if key == 32 or corrected is not None:
                if command_pose is not None:
                    correct_command_index = 0
                    state = STATE_CORRECT_COMMAND
                else:
                    reset_static()

        elif state == STATE_CORRECT_COMMAND:
            if key == ord("4"):
                correct_command_index = (correct_command_index - 1) % len(COMMANDS)
            elif key == ord("6"):
                correct_command_index = (correct_command_index + 1) % len(COMMANDS)
            elif key in (ord("5"), 13):
                corrected = COMMANDS[correct_command_index]
                n = confirm_static_sample(corrected, command_pose, COMMANDS_DATA_PATH, RAW_COMMANDS_DATA_PATH)
                command_signatures = load_signatures(COMMANDS_DATA_PATH, "comandos")
                stats["manual_corrigidas"] += 1
                set_feedback(f"[Corrigido manual] comando -> '{corrected}' ({n} amostras)")
                reset_static()
            elif key == 32:
                reset_static()

        elif key == ord("s") and state == STATE_MANUAL_CONFIRM_MOVEMENT:
            name = movement_ranking[0][0]
            n = confirm_movement_sample(name, movement_sequence)
            movement_signatures = dyn.load_movement_signatures(MOVEMENT_DATA_PATH)
            stats["manual_corretas"] += 1
            set_feedback(f"[OK manual] Movimento '{name}' confirmado ({n} amostras)")
            reset_movement()

        elif key == ord("n") and state == STATE_MANUAL_CONFIRM_MOVEMENT:
            stats["manual_erradas"] += 1
            state = STATE_CORRECT_MOVEMENT

        elif state == STATE_CORRECT_MOVEMENT:
            if key == 32:
                reset_movement()
            elif 32 <= key <= 126:
                ch = chr(key).upper()
                if ch in movement_letters:
                    n = confirm_movement_sample(ch, movement_sequence)
                    movement_signatures = dyn.load_movement_signatures(MOVEMENT_DATA_PATH)
                    stats["manual_corrigidas"] += 1
                    set_feedback(f"[Corrigido manual] movimento -> '{ch}' ({n} amostras)")
                    reset_movement()

        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nSessão encerrada. {stats}")


if __name__ == "__main__":
    main()
