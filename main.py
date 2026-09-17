"""
LIBRAS - Runner Principal
------------------------------
Menu único pra não precisar decorar comando nenhum. Detecta
automaticamente a pasta nn/ (LIBRAS NN Platform) ao lado deste
arquivo e passa --nn-dir sozinho pros scripts que precisam dela.

Antes de rodar qualquer opção, confere se os arquivos que ela precisa
existem (módulos .py, hand_landmarker.task, checkpoints treinados,
etc.) e avisa de forma clara em vez de deixar o script quebrar com um
traceback confuso no meio.

Uso:
    python main.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NN_DIR = os.path.join(BASE_DIR, "nn")
CHECKPOINTS_DIR = os.path.join(NN_DIR, "checkpoints")

PY = sys.executable  # garante o MESMO interpretador Python que rodou main.py


def has_file(*names):
    return all(os.path.exists(os.path.join(BASE_DIR, n)) for n in names)


def has_checkpoint(prefix):
    return os.path.exists(os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest.pt"))


def run(cmd, cwd=BASE_DIR):
    print(f"\n$ {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=cwd, check=False)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    except FileNotFoundError as e:
        print(f"\nERRO ao tentar rodar: {e}")


def pause():
    input("\nPressione ENTER para voltar ao menu...")


# --------------------------------------------------------------------------
# Cada opção: (texto do menu, função de checagem de pré-requisito -> msg de erro ou None, ação)
# --------------------------------------------------------------------------

def check_capture_signatures():
    if not has_file("dynamic_signatures.py", "matching_improved.py"):
        return "Falta dynamic_signatures.py e/ou matching_improved.py na pasta raiz."
    return None


def check_photo_capture():
    return None  # sem dependências especiais


def check_batch_train():
    if not os.path.isdir(os.path.join(BASE_DIR, "imgs")):
        return "Pasta 'imgs/' não encontrada -- crie imgs/<LETRA>/ com fotos antes de rodar."
    return None


def check_translator_old():
    if not has_file("matching_improved.py"):
        return "Falta matching_improved.py na pasta raiz."
    if not has_file("data.json"):
        return "Falta data.json -- capture letras primeiro (opção de captura)."
    return None


def check_auto_validador_ia():
    if not has_file("matching_improved.py", "dynamic_signatures.py", "ai_judge.py", "ollama_config.py"):
        return "Falta matching_improved.py, dynamic_signatures.py, ai_judge.py ou ollama_config.py."
    return None


def check_translator_nn():
    if not has_checkpoint("model_static"):
        return ("Nenhum checkpoint treinado ainda (nn/checkpoints/model_static_latest.pt). "
                "Rode a opção de treino primeiro.")
    return None


def check_auto_validador_nn():
    if not has_file("dynamic_signatures.py"):
        return "Falta dynamic_signatures.py na pasta raiz (usado pra gravar gestos de movimento)."
    return None


def check_train():
    if not has_file("data_raw.json") and not os.path.isdir(os.path.join(BASE_DIR, "imgs")):
        return "Nenhum dado ainda: falta data_raw.json e/ou a pasta imgs/. Capture algo primeiro."
    return None


def check_daemon():
    return check_train()


def check_monitoring():
    return None  # só avisamos sobre Docker na hora, não bloqueamos


# --- Ações ---

def action_capture_signatures():
    run([PY, "capture_signatures.py"])


def action_photo_capture():
    run([PY, "photo_capture.py"])


def action_batch_train():
    run([PY, "batch_train_from_images.py"])


def action_translator_old():
    run([PY, "translator.py"])


def action_auto_validador_ia():
    run([PY, "auto_validador_ia.py"])


def action_translator_nn():
    run([PY, "translator_nn.py", "--nn-dir", NN_DIR])


def action_auto_validador_nn():
    run([PY, "auto_validador_nn.py", "--nn-dir", NN_DIR])


def action_train():
    data_raw = os.path.join(BASE_DIR, "data_raw.json")
    commands_raw = os.path.join(BASE_DIR, "commands_raw.json")
    movement_raw = os.path.join(BASE_DIR, "movement_raw.json")
    imgs_dir = os.path.join(BASE_DIR, "imgs")
    cmd = [
        PY, "train.py",
        "--data-raw", data_raw,
        "--commands-raw", commands_raw,
        "--movement-raw", movement_raw,
        "--checkpoints-dir", CHECKPOINTS_DIR,
    ]
    if os.path.isdir(imgs_dir):
        cmd += ["--imgs-dir", imgs_dir]
    run(cmd, cwd=os.path.join(NN_DIR, "train"))


def action_daemon():
    data_raw = os.path.join(BASE_DIR, "data_raw.json")
    commands_raw = os.path.join(BASE_DIR, "commands_raw.json")
    movement_raw = os.path.join(BASE_DIR, "movement_raw.json")
    print("\nO daemon fica rodando até você fechar esta janela (Ctrl+C pra parar).")
    print("Deixe capture_signatures.py rodando em OUTRA janela enquanto isso.\n")
    run([
        PY, "train_daemon.py",
        "--watch-dir", BASE_DIR,
        "--checkpoints-dir", CHECKPOINTS_DIR,
    ], cwd=os.path.join(NN_DIR, "train"))


def action_monitoring():
    monitoring_dir = os.path.join(NN_DIR, "monitoring")
    print(f"\nIsso requer Docker instalado. Rodando 'docker compose up -d' em:\n  {monitoring_dir}\n")
    run(["docker", "compose", "up", "-d"], cwd=monitoring_dir)
    print("\nSe funcionou: Grafana em http://localhost:3000 (login admin/admin), Prometheus em http://localhost:9090")


def check_panel():
    panel_app = os.path.join(BASE_DIR, "panel", "app.py")
    if not os.path.exists(panel_app):
        return "panel/app.py não encontrado."
    return None


def action_panel():
    print("\nAbrindo o painel em http://localhost:5000 -- deixe esta janela aberta.")
    run([PY, "app.py"], cwd=os.path.join(BASE_DIR, "panel"))


MENU = [
    ("1", "Capturar assinaturas (letras/comandos/movimento) -- webcam", check_capture_signatures, action_capture_signatures),
    ("2", "Capturar fotos pra imgs/ -- webcam", check_photo_capture, action_photo_capture),
    ("3", "Treinar a partir de imgs/ (k-NN antigo, gera data.json)", check_batch_train, action_batch_train),
    ("4", "Tradutor em tempo real -- k-NN antigo (translator.py)", check_translator_old, action_translator_old),
    ("5", "Auto-validador -- Ollama antigo (auto_validador_ia.py)", check_auto_validador_ia, action_auto_validador_ia),
    ("6", "Treinar a rede neural (train.py)", check_train, action_train),
    ("7", "Tradutor em tempo real -- rede neural (translator_nn.py)", check_translator_nn, action_translator_nn),
    ("8", "Auto-validador -- rede neural (auto_validador_nn.py)", check_auto_validador_nn, action_auto_validador_nn),
    ("9", "Ligar o daemon de treino contínuo (train_daemon.py)", check_daemon, action_daemon),
    ("10", "Subir Grafana + Prometheus (requer Docker)", check_monitoring, action_monitoring),
    ("11", "Abrir painel de controle web (sem terminal, http://localhost:5000)", check_panel, action_panel),
    ("0", "Sair", None, None),
]


def print_menu():
    print("\n" + "=" * 60)
    print("LIBRAS - Menu Principal")
    print("=" * 60)
    for key, label, check, _ in MENU:
        if key == "0":
            print(f"  {key}. {label}")
            continue
        warn = ""
        if check is not None:
            problem = check()
            if problem:
                warn = f"   [BLOQUEADO: {problem}]"
        print(f"  {key}. {label}{warn}")
    print("=" * 60)


def main():
    while True:
        print_menu()
        choice = input("Escolha uma opção: ").strip()
        entry = next((m for m in MENU if m[0] == choice), None)
        if entry is None:
            print("Opção inválida.")
            continue
        key, label, check, action = entry
        if key == "0":
            print("Até mais!")
            break
        if check is not None:
            problem = check()
            if problem:
                print(f"\nNão dá pra rodar essa opção ainda: {problem}")
                pause()
                continue
        action()
        pause()


if __name__ == "__main__":
    main()
