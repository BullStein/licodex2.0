"""
Painel de Controle - LIBRAS
--------------------------------
Interface web local pra gerenciar captura/treino sem precisar abrir
terminal nenhum: botões pra iniciar capture_signatures.py,
photo_capture.py, rodar o treino (train.py) e ligar/desligar o
daemon contínuo (train_daemon.py). Mostra status (últimas métricas de
cada checkpoint, se o daemon está rodando) e o log ao vivo de cada
processo.

IMPORTANTE sobre os scripts de webcam (capture_signatures.py,
photo_capture.py): eles abrem sua PRÓPRIA janela OpenCV (não dá pra
mostrar webcam dentro do navegador sem reescrever aqueles scripts do
zero) -- o painel só clica o botão de "iniciar" pra você, a janela
abre separada, como se você tivesse rodado no terminal.

Uso:
    python app.py
    (abre em http://localhost:5000)
"""

import os
import sys
import json
import subprocess
import threading

from flask import Flask, jsonify, render_template, request

PANEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PANEL_DIR)          # raiz do projeto (libras-project/)
NN_DIR = os.path.join(BASE_DIR, "nn")
TRAIN_DIR = os.path.join(NN_DIR, "train")
CHECKPOINTS_DIR = os.path.join(NN_DIR, "checkpoints")
LOG_DIR = os.path.join(PANEL_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

PY = sys.executable

app = Flask(__name__)

# --------------------------------------------------------------------------
# Gerenciamento de processos em background (um por "ação")
# --------------------------------------------------------------------------

_lock = threading.Lock()
_processes = {}  # nome -> subprocess.Popen


def _log_path(name):
    return os.path.join(LOG_DIR, f"{name}.log")


def _is_running(name):
    proc = _processes.get(name)
    return proc is not None and proc.poll() is None


def _start(name, cmd, cwd):
    with _lock:
        if _is_running(name):
            return False, "já está rodando"
        log_file = open(_log_path(name), "w", encoding="utf-8")
        # -u (unbuffered) é essencial aqui: sem isso, o Python bufferiza a
        # saída inteira até o processo terminar quando stdout não é um
        # terminal (como é o caso, redirecionado pro arquivo de log) --
        # o log ao vivo no painel ficaria vazio até o fim do processo.
        unbuffered_cmd = [cmd[0], "-u"] + cmd[1:]
        try:
            proc = subprocess.Popen(
                unbuffered_cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as e:
            log_file.write(f"ERRO ao iniciar: {e}\n")
            log_file.close()
            return False, str(e)
        _processes[name] = proc
        return True, None


def _stop(name):
    with _lock:
        proc = _processes.get(name)
        if proc is None or proc.poll() is not None:
            return False, "não está rodando"
        proc.terminate()
        return True, None


def _tail_log(name, n=200):
    path = _log_path(name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return lines[-n:]


# --------------------------------------------------------------------------
# Leitura de status (checkpoints, dados capturados)
# --------------------------------------------------------------------------

def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _checkpoint_status(prefix):
    meta = _read_json(os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest.meta.json"))
    if meta is None:
        return {"trained": False}
    return {
        "trained": True,
        "val_accuracy": meta.get("val_accuracy"),
        "val_loss": meta.get("val_loss"),
        "num_classes": meta.get("num_classes"),
        "num_samples": meta.get("num_samples"),
        "trained_at": meta.get("trained_at"),
        "version": meta.get("version"),
    }


def _count_raw_samples(path):
    data = _read_json(os.path.join(BASE_DIR, path))
    if not data:
        return {"classes": 0, "samples": 0}
    total = 0
    for v in data.values():
        # cada entrada é uma pose única (21,3) ou uma lista de poses
        try:
            total += len(v) if isinstance(v[0], list) and isinstance(v[0][0], list) else 1
        except (IndexError, TypeError):
            total += 1
    return {"classes": len(data), "samples": total}


def _imgs_status():
    imgs_dir = os.path.join(BASE_DIR, "imgs")
    if not os.path.isdir(imgs_dir):
        return {"exists": False, "folders": 0, "images": 0}
    folders = [d for d in os.listdir(imgs_dir) if os.path.isdir(os.path.join(imgs_dir, d))]
    total_images = 0
    for d in folders:
        total_images += len([
            f for f in os.listdir(os.path.join(imgs_dir, d))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
    return {"exists": True, "folders": len(folders), "images": total_images}


# --------------------------------------------------------------------------
# Rotas
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "processes": {name: _is_running(name) for name in
                       ["capture_signatures", "photo_capture", "train", "daemon", "sync"]},
        "checkpoints": {
            "static": _checkpoint_status("model_static"),
            "movement": _checkpoint_status("model_movement"),
        },
        "data": {
            "letters": _count_raw_samples("data_raw.json"),
            "commands": _count_raw_samples("commands_raw.json"),
            "movement": _count_raw_samples("movement_raw.json"),
            "imgs": _imgs_status(),
        },
    })


@app.route("/log/<name>")
def log(name):
    if name not in ("capture_signatures", "photo_capture", "train", "daemon", "sync"):
        return jsonify({"error": "nome inválido"}), 400
    return jsonify({"lines": _tail_log(name)})


@app.route("/action/capture_signatures", methods=["POST"])
def action_capture_signatures():
    ok, err = _start("capture_signatures", [PY, "capture_signatures.py"], BASE_DIR)
    return jsonify({"ok": ok, "error": err})


@app.route("/action/photo_capture", methods=["POST"])
def action_photo_capture():
    ok, err = _start("photo_capture", [PY, "photo_capture.py"], BASE_DIR)
    return jsonify({"ok": ok, "error": err})


@app.route("/action/train", methods=["POST"])
def action_train():
    imgs_dir = os.path.join(BASE_DIR, "imgs")
    cmd = [
        PY, "train.py",
        "--data-raw", os.path.join(BASE_DIR, "data_raw.json"),
        "--commands-raw", os.path.join(BASE_DIR, "commands_raw.json"),
        "--movement-raw", os.path.join(BASE_DIR, "movement_raw.json"),
        "--checkpoints-dir", CHECKPOINTS_DIR,
    ]
    if os.path.isdir(imgs_dir):
        cmd += ["--imgs-dir", imgs_dir]
    ok, err = _start("train", cmd, TRAIN_DIR)
    return jsonify({"ok": ok, "error": err})


@app.route("/action/daemon/start", methods=["POST"])
def action_daemon_start():
    cmd = [
        PY, "train_daemon.py",
        "--watch-dir", BASE_DIR,
        "--checkpoints-dir", CHECKPOINTS_DIR,
    ]
    ok, err = _start("daemon", cmd, TRAIN_DIR)
    return jsonify({"ok": ok, "error": err})


@app.route("/action/daemon/stop", methods=["POST"])
def action_daemon_stop():
    ok, err = _stop("daemon")
    return jsonify({"ok": ok, "error": err})


@app.route("/action/sync", methods=["POST"])
def action_sync():
    sync_config = os.path.join(BASE_DIR, "sync_config.json")
    if not os.path.exists(sync_config):
        return jsonify({"ok": False, "error": "sync_config.json não encontrado na raiz do projeto -- "
                                                "veja sync_server/README.md pra criar."})
    ok, err = _start("sync", [PY, "sync_client.py"], BASE_DIR)
    return jsonify({"ok": ok, "error": err})


if __name__ == "__main__":
    print(f"Painel em http://localhost:5000  (BASE_DIR={BASE_DIR})")
    app.run(host="127.0.0.1", port=5000, debug=False)
