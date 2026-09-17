"""
Servidor de Sincronização - LIBRAS
--------------------------------------
API pra sincronizar dados entre vários PCs (ex.: pessoal + faculdade)
sem perder amostras capturadas em cada máquina. Pensado pra hospedar
num serviço externo (Render, Railway, VPS) -- veja o README ao lado
pra instruções de deploy.

DECISÃO DE DESIGN: merge, não overwrite.
    Se cada PC simplesmente "subisse" seu data_raw.json por cima do
    que já está no servidor, a última máquina a sincronizar apagaria
    o que a outra tinha capturado sozinha. Em vez disso:

    - JSONs brutos (data_raw/commands_raw/movement_raw): o servidor
      faz UNIÃO por rótulo (letra/comando) entre o que chegou e o que
      já tinha, com deduplicação exata -- nunca perde amostra, nunca
      duplica a mesma amostra duas vezes.
    - Imagens (imgs/): cada arquivo tem nome único; em caso de
      colisão (dois PCs geraram "A_0001.jpg" com fotos diferentes),
      o servidor renomeia automaticamente em vez de sobrescrever.
    - Checkpoints (.pt): só substitui o "latest" se a acurácia de
      validação for >= a que já estava lá -- mesma lógica de
      promoção do train.py, aplicada agora entre máquinas.

Autenticação: um token fixo (var de ambiente SYNC_API_TOKEN),
enviado no header "Authorization: Bearer <token>". Sem isso qualquer
um que ache a URL do seu servidor poderia escrever/ler seus dados.

Variáveis de ambiente:
    SYNC_API_TOKEN   (obrigatório) -- token que os clientes devem enviar
    SYNC_STORAGE_DIR (opcional, padrão "./storage") -- onde os arquivos
                      ficam salvos no servidor. Em produção, aponte
                      isso pra um volume PERSISTENTE (ver README) --
                      sem isso, um redeploy apaga tudo.

Uso local (teste):
    set SYNC_API_TOKEN=segredo123
    python app.py
"""

import os
import io
import json
import time
import hashlib
from functools import wraps

from flask import Flask, request, jsonify, send_file, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("SYNC_STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
IMGS_DIR = os.path.join(STORAGE_DIR, "imgs")
CHECKPOINTS_DIR = os.path.join(STORAGE_DIR, "checkpoints")
os.makedirs(IMGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

API_TOKEN = os.environ.get("SYNC_API_TOKEN")

RAW_JSON_NAMES = {"data_raw", "commands_raw", "movement_raw"}
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

app = Flask(__name__)


# --------------------------------------------------------------------------
# Autenticação
# --------------------------------------------------------------------------

def require_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not API_TOKEN:
            return jsonify({"error": "Servidor mal configurado: SYNC_API_TOKEN não definido."}), 500
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Token inválido ou ausente."}), 401
        return f(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------
# Utilidades de JSON bruto (mesmo formato do capture_signatures.py)
# --------------------------------------------------------------------------

def _json_path(name):
    return os.path.join(STORAGE_DIR, f"{name}.json")


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _sample_key(sample):
    """Chave estável pra deduplicar uma amostra (pose ou sequência),
    independente de reordenação -- arredonda antes de fazer o hash
    pra tolerar diferenças de ponto flutuante insignificantes entre
    o que foi salvo e recarregado em máquinas diferentes."""
    def round_nested(x):
        if isinstance(x, list):
            return [round_nested(v) for v in x]
        return round(x, 6)
    normalized = round_nested(sample)
    return hashlib.sha256(json.dumps(normalized).encode("utf-8")).hexdigest()


def _merge_pool(existing, incoming):
    """existing, incoming: dict rótulo -> lista de amostras.
    Retorna a união deduplicada, por rótulo."""
    merged = {k: list(v) for k, v in existing.items()}
    added = 0
    for label, samples in incoming.items():
        current = merged.setdefault(label, [])
        seen = {_sample_key(s) for s in current}
        for s in samples:
            key = _sample_key(s)
            if key not in seen:
                current.append(s)
                seen.add(key)
                added += 1
    return merged, added


# --------------------------------------------------------------------------
# Rotas: JSONs brutos (merge)
# --------------------------------------------------------------------------

@app.route("/merge/<name>", methods=["POST"])
@require_token
def merge_json(name):
    if name not in RAW_JSON_NAMES:
        return jsonify({"error": f"nome inválido, use um de {sorted(RAW_JSON_NAMES)}"}), 400

    incoming = request.get_json(silent=True)
    if incoming is None or not isinstance(incoming, dict):
        return jsonify({"error": "corpo precisa ser um JSON objeto {rótulo: [amostras...]}"}), 400

    path = _json_path(name)
    existing = _load_json(path)
    merged, added = _merge_pool(existing, incoming)
    _save_json(path, merged)

    return jsonify({
        "ok": True,
        "added": added,
        "total_classes": len(merged),
        "total_samples": sum(len(v) for v in merged.values()),
        "merged": merged,  # devolve o estado final pra o cliente sobrescrever o arquivo local
    })


@app.route("/download/<name>", methods=["GET"])
@require_token
def download_json(name):
    if name not in RAW_JSON_NAMES:
        return jsonify({"error": f"nome inválido, use um de {sorted(RAW_JSON_NAMES)}"}), 400
    return jsonify(_load_json(_json_path(name)))


# --------------------------------------------------------------------------
# Rotas: imagens (imgs/<label>/<filename>)
# --------------------------------------------------------------------------

@app.route("/imgs/manifest", methods=["GET"])
@require_token
def imgs_manifest():
    """Devolve {label: [{"filename": ..., "sha256": ...}, ...]}. O hash é o
    que importa pra decidir se um arquivo já existe -- o NOME sozinho não
    é confiável entre PCs diferentes (cada um numera A_0001.jpg, A_0002.jpg
    de forma independente, então nomes iguais podem ser fotos diferentes,
    e fotos idênticas podem ter nomes diferentes)."""
    manifest = {}
    if os.path.isdir(IMGS_DIR):
        for label in sorted(os.listdir(IMGS_DIR)):
            label_dir = os.path.join(IMGS_DIR, label)
            if os.path.isdir(label_dir):
                entries = []
                for f in sorted(os.listdir(label_dir)):
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        fpath = os.path.join(label_dir, f)
                        with open(fpath, "rb") as fh:
                            digest = hashlib.sha256(fh.read()).hexdigest()
                        entries.append({"filename": f, "sha256": digest})
                manifest[label] = entries
    return jsonify(manifest)


@app.route("/imgs/upload", methods=["POST"])
@require_token
def imgs_upload():
    label = request.form.get("label")
    filename = request.form.get("filename")
    file = request.files.get("file")
    if not label or not filename or file is None:
        return jsonify({"error": "form precisa de 'label', 'filename' e 'file'"}), 400
    if not filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
        return jsonify({"error": "extensão de arquivo não suportada"}), 400

    label_dir = os.path.join(IMGS_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    dest = os.path.join(label_dir, filename)
    if os.path.exists(dest):
        # colisão de nome entre PCs diferentes -- nunca sobrescreve,
        # renomeia com sufixo incremental até achar um nome livre.
        stem, ext = os.path.splitext(filename)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(label_dir, f"{stem}_dup{n}{ext}")
            n += 1

    file.save(dest)
    return jsonify({"ok": True, "saved_as": os.path.basename(dest)})


@app.route("/imgs/download/<label>/<filename>", methods=["GET"])
@require_token
def imgs_download(label, filename):
    path = os.path.join(IMGS_DIR, label, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


# --------------------------------------------------------------------------
# Rotas: checkpoints (.pt) -- só promove se val_accuracy >= o que já tem
# --------------------------------------------------------------------------

@app.route("/checkpoints/<prefix>/meta", methods=["GET"])
@require_token
def checkpoint_meta(prefix):
    meta_path = os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest.meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"trained": False})
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return jsonify({"trained": True, **meta})


@app.route("/checkpoints/<prefix>/upload", methods=["POST"])
@require_token
def checkpoint_upload(prefix):
    meta_raw = request.form.get("meta")
    model_file = request.files.get("model")
    labels_file = request.files.get("labels")
    if not meta_raw or model_file is None or labels_file is None:
        return jsonify({"error": "form precisa de 'meta' (JSON), 'model' (.pt) e 'labels' (.json)"}), 400

    try:
        new_meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        return jsonify({"error": "'meta' não é um JSON válido"}), 400

    meta_path = os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest.meta.json")
    current_meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            current_meta = json.load(f)

    new_acc = new_meta.get("val_accuracy")
    current_acc = current_meta.get("val_accuracy") if current_meta else None

    if current_meta is not None and (new_acc is None or (current_acc is not None and new_acc < current_acc)):
        return jsonify({
            "ok": True, "promoted": False,
            "reason": f"val_accuracy enviada ({new_acc}) menor que a já armazenada ({current_acc})",
        })

    model_file.save(os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest.pt"))
    labels_file.save(os.path.join(CHECKPOINTS_DIR, f"{prefix}_latest_labels.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "promoted": True})


@app.route("/checkpoints/<prefix>/download/<kind>", methods=["GET"])
@require_token
def checkpoint_download(prefix, kind):
    filenames = {
        "model": f"{prefix}_latest.pt",
        "labels": f"{prefix}_latest_labels.json",
        "meta": f"{prefix}_latest.meta.json",
    }
    if kind not in filenames:
        return jsonify({"error": "kind precisa ser 'model', 'labels' ou 'meta'"}), 400
    path = os.path.join(CHECKPOINTS_DIR, filenames[kind])
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


# --------------------------------------------------------------------------
# Saúde
# --------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "time": time.time()})


if __name__ == "__main__":
    if not API_TOKEN:
        print("AVISO: SYNC_API_TOKEN não definido -- todas as rotas autenticadas vão retornar 500.")
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
