"""
Tradutor LIBRAS -> Letras (versão Rede Neural)
------------------------------------------------
Equivalente ao translator.py original, mas usa a LIBRAS NN Platform
(StaticSignNet + MovementSignNet, via inference/predictor.py) no
lugar do matching_improved.py/dynamic_signatures.py (k-NN).

O QUE MUDOU em relação ao translator.py original:
    - classify_hand() não usa mais classify_ranked_weighted/is_confident
      -- chama predictor.predict_static() direto, que já devolve a
      classe prevista + confiança calibrada pelo softmax da rede.
    - O limiar de aceitação agora é só um número de confiança
      (--confidence-threshold), em vez de threshold de distância +
      margem mínima -- mais simples porque a rede já concentra tudo
      numa única métrica de confiança.
    - Opcionalmente expõe métricas de inferência (confiança, latência)
      via metrics/exporter.py, pra aparecer no Grafana junto com as
      métricas de treino do train_daemon.py.
    - NOVO: reconhecimento de MOVIMENTO em tempo real (H, J, K, X, Z...).
      Nem o translator.py original tinha isso -- ele só carregava
      data.json/commands.json, nunca movement.json. Funciona por
      detecção automática de início de gesto: usa
      dynamic_signatures.motion_energy() pra perceber quando a mão
      esquerda começa a se mover acima de um limiar, grava
      --record-seconds de janela sozinho, classifica com
      MovementSignNet ao final (reamostrando pra
      dynamic_signatures.DEFAULT_SEQUENCE_LENGTH frames, o mesmo
      tamanho fixo que capture_signatures.py sempre usou pra gerar os
      dados de treino), e só então volta a aceitar letras estáticas.
      Não precisa apertar tecla nenhuma. Ative com --enable-movement
      (requer nn/checkpoints/model_movement_latest.pt já treinado --
      se não existir, o script avisa e segue só com letra estática,
      sem travar).
    - Mesma UI, mesmos painéis, mesmo histórico -- só troca o "motor"
      de classificação por baixo.

Pré-requisitos:
    - Ter rodado train.py pelo menos uma vez (checkpoints promovidos
      em nn/checkpoints/)
    - Rodar este script na MESMA pasta que hand_landmarker.task e
      dynamic_signatures.py (ele reusa o mesmo model_asset_path e a
      mesma lógica de motion_energy/resample_sequence dos outros
      scripts do projeto)

Como usar:
    python translator_nn.py --nn-dir nn
    python translator_nn.py --nn-dir nn --confidence-threshold 0.7
    python translator_nn.py --nn-dir nn --enable-movement
    python translator_nn.py --nn-dir nn --enable-movement --motion-threshold 1.2 --record-seconds 1.0
    python translator_nn.py --nn-dir nn --metrics-port 9309   (ativa métricas Prometheus)

Controles:
    - "0" ou ESC para sair
    - "9" para limpar o histórico manualmente
"""

import os
import sys
import math
import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "Tradutor LIBRAS (Rede Neural)"
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 620

TARGET_FPS = 60
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

LETTER_HAND_LABEL = "Left"
COMMAND_HAND_LABEL = "Right"

WRIST = 0
MIDDLE_MCP = 9

HISTORY_MAX_LEN = 30
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]

# Quantos frames recentes olhar pra decidir "a mão começou a se mover
# agora" -- uma janela curta, só pra pegar o início do gesto, não o
# gesto inteiro (esse vem depois, na gravação de --record-seconds).
MOTION_TRIGGER_WINDOW = 5

# Quanto tempo (segundos) o resultado de um gesto de movimento fica
# exibido no painel depois de classificado, antes de voltar a mostrar
# a letra estática normalmente.
MOVEMENT_RESULT_DISPLAY_SECONDS = 1.5


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def normalize_landmarks(landmarks):
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array([[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
                      for lm in landmarks])


def classify_hand(predictor, normalized_pose, confidence_threshold, restrict_to=None):
    """
    Substitui o classify_hand do translator.py original: chama a rede
    direto, sem k-NN. `restrict_to` (opcional) filtra a predição pra só
    aceitar se a sugestão estiver num conjunto específico de rótulos --
    usado pra mão de comando, já que ela usa a MESMA rede das letras
    (treinadas juntas) mas só deve reagir a BACKSPACE/SPACE/CLEAR.
    """
    if predictor is None or not predictor.static_ready:
        return None, 0.0
    veredito = predictor.predict_static(normalized_pose)
    sugestao, confianca = veredito["sugestao"], veredito["confianca"]
    if confianca < confidence_threshold:
        return None, confianca
    if restrict_to is not None and sugestao not in restrict_to:
        return None, confianca
    return sugestao, confianca


def smooth_prediction(buffer, new_value, maxlen=8):
    buffer.append(new_value)
    if len(buffer) > maxlen:
        buffer.popleft()
    most_common, _ = Counter(buffer).most_common(1)[0]
    return most_common


def draw_transparent_box(frame, top_left, bottom_right, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def apply_command(command, letter_history):
    if command == "BACKSPACE":
        return letter_history[:-1]
    if command == "SPACE":
        new_history = letter_history + " "
        return new_history[-HISTORY_MAX_LEN:]
    if command == "CLEAR":
        return ""
    return letter_history


def main():
    parser = argparse.ArgumentParser(description="Tradutor LIBRAS em tempo real usando a rede neural.")
    parser.add_argument("--nn-dir", required=True,
                         help="Caminho pra pasta nn/ (contém inference/predictor.py e train/).")
    parser.add_argument("--checkpoints-dir", default=None,
                         help="Padrão: <nn-dir>/checkpoints")
    parser.add_argument("--confidence-threshold", type=float, default=0.6,
                         help="Confiança mínima pra aceitar uma predição (padrão: 0.6). "
                              "Baixe se a rede estiver 'travada' demais, suba se estiver trocando letras.")
    parser.add_argument("--metrics-port", type=int, default=None,
                         help="Se passado, expõe métricas Prometheus de inferência nessa porta (ex.: 9309).")
    parser.add_argument("--enable-movement", action="store_true",
                         help="Ativa reconhecimento de movimento em tempo real (H, J, K, X, Z...). "
                              "Requer model_movement_latest.pt já treinado.")
    parser.add_argument("--motion-threshold", type=float, default=1.0,
                         help="Energia de movimento mínima (motion_energy) pra considerar que a mão "
                              "começou um gesto de movimento (padrão: 1.0 -- ajuste observando sua webcam).")
    parser.add_argument("--record-seconds", type=float, default=1.0,
                         help="Duração da janela gravada quando um gesto de movimento é detectado (padrão: 1.0s).")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.nn_dir, "inference"))
    sys.path.insert(0, os.path.join(args.nn_dir, "train"))
    from predictor import SignPredictor

    checkpoints_dir = args.checkpoints_dir or os.path.join(args.nn_dir, "checkpoints")
    predictor = SignPredictor(checkpoints_dir)
    print(f"Rede carregada -> static_ready={predictor.static_ready} (checkpoints: {checkpoints_dir})")
    if not predictor.static_ready:
        print("AVISO: nenhum checkpoint de produção encontrado ainda. "
              "Rode train.py primeiro (ver train/train.py).")

    movement_enabled = args.enable_movement
    dyn = None
    if movement_enabled:
        if not predictor.movement_ready:
            print("AVISO: --enable-movement pedido, mas não há checkpoint de movimento treinado ainda "
                  "(model_movement_latest.pt). Reconhecimento de movimento fica DESLIGADO por enquanto.")
            movement_enabled = False
        else:
            try:
                import dynamic_signatures as dyn
            except ImportError:
                print("AVISO: --enable-movement pedido, mas dynamic_signatures.py não foi encontrado "
                      "na pasta deste script. Reconhecimento de movimento fica DESLIGADO.")
                movement_enabled = False
    if movement_enabled:
        print(f"Reconhecimento de MOVIMENTO ativo -> limiar={args.motion_threshold}, "
              f"janela={args.record_seconds:.1f}s, sequence_length={dyn.DEFAULT_SEQUENCE_LENGTH}")

    metrics = None
    if args.metrics_port:
        sys.path.insert(0, os.path.join(args.nn_dir, "metrics"))
        from exporter import InferenceMetrics
        metrics = InferenceMetrics(port=args.metrics_port)

    # Letras conhecidas pela rede (pra separar do que é comando, já que
    # ambos compartilham a mesma rede/label_encoder)
    known_labels = set(predictor._static.label_encoder.classes) if predictor.static_ready else set()
    known_letters = known_labels - set(COMMANDS)
    known_commands = known_labels & set(COMMANDS)
    available_commands = sorted(known_commands)
    known_movement_letters = (
        set(predictor._movement.label_encoder.classes) if movement_enabled and predictor.movement_ready else set()
    )

    ensure_model()

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

    letter_buffer = deque(maxlen=8)
    command_buffer = deque(maxlen=6)
    frame_timestamp_ms = 0
    start_time = time.time()

    letter_history = ""
    last_committed_letter = None
    last_committed_command = None
    last_command_seen = "-"
    last_letter_confidence = 0.0

    # --- Estado do reconhecimento de movimento (só usado se movement_enabled) ---
    recent_poses = deque(maxlen=MOTION_TRIGGER_WINDOW)  # janela curta, só pra detectar o "começo" do gesto
    is_recording_movement = False
    recording_buffer = []
    recording_start = 0.0
    last_movement_result = None   # (letra, confiança) da última classificação de movimento, pra exibir um instante
    last_movement_shown_until = 0.0

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

        letter = "-"
        raw_command = None
        live_letter_pose = None

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name

                if label == LETTER_HAND_LABEL:
                    live_letter_pose = normalize_landmarks(hand_landmarks)

                elif label == COMMAND_HAND_LABEL:
                    normalized_pose = normalize_landmarks(hand_landmarks)
                    raw_command, _ = classify_hand(
                        predictor, normalized_pose, args.confidence_threshold, restrict_to=known_commands,
                    )

        # --- Máquina de estados do movimento (só entra em jogo se ativado) ---
        if movement_enabled and is_recording_movement:
            if live_letter_pose is not None:
                recording_buffer.append(live_letter_pose)

            if time.time() - recording_start >= args.record_seconds:
                is_recording_movement = False
                if len(recording_buffer) >= 2:
                    sequence = dyn.resample_sequence(recording_buffer, dyn.DEFAULT_SEQUENCE_LENGTH)
                    if metrics:
                        with metrics.time_prediction("movement"):
                            mv_veredito = predictor.predict_movement(sequence)
                        metrics.record_confidence("movement", mv_veredito["confianca"])
                    else:
                        mv_veredito = predictor.predict_movement(sequence)

                    print(f"[movimento] {mv_veredito}")
                    if mv_veredito["veredito"] == "correto" and mv_veredito["sugestao"] in known_movement_letters:
                        mv_letter = mv_veredito["sugestao"]
                        letter_history += mv_letter
                        letter_history = letter_history[-HISTORY_MAX_LEN:]
                        last_movement_result = (mv_letter, mv_veredito["confianca"])
                        last_movement_shown_until = time.time() + MOVEMENT_RESULT_DISPLAY_SECONDS
                recording_buffer = []

        elif movement_enabled and live_letter_pose is not None:
            recent_poses.append(live_letter_pose)
            if len(recent_poses) == MOTION_TRIGGER_WINDOW:
                energy = dyn.motion_energy(np.array(recent_poses))
                if energy >= args.motion_threshold:
                    is_recording_movement = True
                    recording_start = time.time()
                    recording_buffer = list(recent_poses)  # já inclui os frames do próprio gatilho
                    recent_poses.clear()

        # --- Letra estática: só roda quando NÃO está gravando movimento,
        #     pra não misturar uma pose "de transição" no histórico ---
        if not is_recording_movement:
            if live_letter_pose is not None:
                if metrics:
                    with metrics.time_prediction("static_letter"):
                        raw_letter, conf = classify_hand(
                            predictor, live_letter_pose, args.confidence_threshold, restrict_to=known_letters,
                        )
                    metrics.record_confidence("static_letter", conf)
                else:
                    raw_letter, conf = classify_hand(
                        predictor, live_letter_pose, args.confidence_threshold, restrict_to=known_letters,
                    )
                last_letter_confidence = conf
                letter = smooth_prediction(letter_buffer, raw_letter or "?")

            if letter not in ("-", "?") and letter != last_committed_letter:
                letter_history += letter
                letter_history = letter_history[-HISTORY_MAX_LEN:]
                last_committed_letter = letter
            elif letter in ("-", "?"):
                last_committed_letter = None

        # --- Comando: suaviza e só dispara quando MUDA ---
        command = smooth_prediction(command_buffer, raw_command if raw_command else "NONE")
        command = None if command == "NONE" else command
        if command:
            last_command_seen = command
        if command is not None and command != last_committed_command:
            letter_history = apply_command(command, letter_history)
            last_committed_command = command
        elif command is None:
            last_committed_command = None

        # --- Overlays ---
        h, w, _ = frame.shape

        draw_transparent_box(frame, (0, 0), (220, 110), color=(0, 0, 0), alpha=0.55)
        if is_recording_movement:
            remaining = max(0.0, args.record_seconds - (time.time() - recording_start))
            cv2.putText(frame, "Movimento...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{remaining:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
        elif last_movement_result and time.time() < last_movement_shown_until:
            mv_letter, mv_conf = last_movement_result
            cv2.putText(frame, "Letra (mov.):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, mv_letter, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"conf: {mv_conf:.2f}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Letra:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, letter, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"conf: {last_letter_confidence:.2f}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 220, 255), 1, cv2.LINE_AA)

        legend_height = 60 + len(available_commands) * 22
        draw_transparent_box(frame, (w - 240, 0), (w, legend_height), color=(0, 0, 0), alpha=0.55)
        cv2.putText(frame, "Comando atual:", (w - 230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, last_command_seen, (w - 230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if available_commands:
            cv2.putText(frame, "Disponiveis:", (w - 230, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            for i, cmd_name in enumerate(available_commands):
                cv2.putText(frame, f"- {cmd_name}", (w - 230, 98 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "(rede sem comandos treinados)", (w - 230, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if movement_enabled:
            cv2.putText(frame, f"Movimento: ON {sorted(known_movement_letters)}", (10, h - 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

        history_box_top = h - 60
        draw_transparent_box(frame, (0, history_box_top), (w, h), color=(0, 0, 0), alpha=0.55)
        cv2.putText(frame, "Historico:", (10, history_box_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        history_text = letter_history if letter_history else "-"
        cv2.putText(frame, history_text, (10, history_box_top + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("0") or key == 27:
            break
        elif key == ord("9"):
            letter_history = ""

        elapsed = time.time() - loop_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
