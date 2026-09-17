"""
Cliente de Sincronização - LIBRAS
--------------------------------------
Roda em CADA PC (pessoal, faculdade, etc.) e conversa com o
sync_server/app.py hospedado externamente. Sobe as amostras novas
capturadas localmente (merge -- nunca perde o que já estava no
servidor nem o que só existe localmente) e baixa o que os OUTROS PCs
já capturaram, pra ficar tudo igual nos dois lados.

Configuração: crie um arquivo sync_config.json ao lado deste script
(ou passe tudo via linha de comando):
    {
      "server_url": "https://seu-servidor.onrender.com",
      "api_token": "o-mesmo-token-que-voce-configurou-no-servidor"
    }

Uso:
    python sync_client.py                     (sincroniza tudo: json + imgs + checkpoints)
    python sync_client.py --only json          (só os _raw.json)
    python sync_client.py --only imgs
    python sync_client.py --only checkpoints
    python sync_client.py --push-only          (só envia, não baixa nada)
    python sync_client.py --pull-only          (só baixa, não envia nada)
"""

import os
import sys
import json
import argparse

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "sync_config.json")

RAW_JSON_NAMES = ["data_raw", "commands_raw", "movement_raw"]
CHECKPOINT_PREFIXES = ["model_static", "model_movement"]
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_config(args):
    server_url = args.server_url
    api_token = args.api_token
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        server_url = server_url or cfg.get("server_url")
        api_token = api_token or cfg.get("api_token")
    if not server_url or not api_token:
        print("ERRO: faltam server_url e/ou api_token. Crie sync_config.json ou use "
              "--server-url e --api-token.")
        sys.exit(1)
    return server_url.rstrip("/"), api_token


def headers(token):
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------
# JSONs brutos
# --------------------------------------------------------------------------

def _load_local_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def sync_json(server_url, token, name, push=True, pull=True):
    path = os.path.join(BASE_DIR, f"{name}.json")
    local = _load_local_json(path)

    if push:
        resp = requests.post(f"{server_url}/merge/{name}", headers=headers(token), json=local, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        print(f"  {name}.json: +{result['added']} amostras novas no servidor "
              f"(total agora: {result['total_classes']} classes, {result['total_samples']} amostras)")
        if pull:
            # o merge já devolve o estado final -- não precisa de uma 2ª chamada
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result["merged"], f, ensure_ascii=False, indent=2)
            print(f"  {name}.json: atualizado localmente com o estado merged")
    elif pull:
        resp = requests.get(f"{server_url}/download/{name}", headers=headers(token), timeout=30)
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resp.json(), f, ensure_ascii=False, indent=2)
        print(f"  {name}.json: baixado do servidor")


# --------------------------------------------------------------------------
# Imagens (imgs/)
# --------------------------------------------------------------------------

def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sync_imgs(server_url, token, push=True, pull=True):
    imgs_dir = os.path.join(BASE_DIR, "imgs")

    if push and os.path.isdir(imgs_dir):
        resp = requests.get(f"{server_url}/imgs/manifest", headers=headers(token), timeout=30)
        resp.raise_for_status()
        remote_manifest = resp.json()  # {label: [{"filename":..., "sha256":...}, ...]}

        uploaded = 0
        for label in sorted(os.listdir(imgs_dir)):
            label_dir = os.path.join(imgs_dir, label)
            if not os.path.isdir(label_dir):
                continue
            # compara por CONTEÚDO (hash), não por nome -- dois PCs numerando
            # A_0001.jpg de forma independente podem ter fotos diferentes com
            # o mesmo nome, ou a mesma foto com nomes diferentes.
            remote_hashes = {entry["sha256"] for entry in remote_manifest.get(label, [])}
            for fname in sorted(os.listdir(label_dir)):
                if not fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(label_dir, fname)
                if _sha256_file(fpath) in remote_hashes:
                    continue  # esse CONTEÚDO já existe no servidor, sob algum nome
                with open(fpath, "rb") as f:
                    r = requests.post(
                        f"{server_url}/imgs/upload", headers=headers(token),
                        data={"label": label, "filename": fname},
                        files={"file": (fname, f)}, timeout=30,
                    )
                r.raise_for_status()
                uploaded += 1
        print(f"  imgs/: {uploaded} foto(s) nova(s) enviada(s)")

    if pull:
        resp = requests.get(f"{server_url}/imgs/manifest", headers=headers(token), timeout=30)
        resp.raise_for_status()
        remote_manifest = resp.json()

        downloaded = 0
        for label, entries in remote_manifest.items():
            label_dir = os.path.join(imgs_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            local_hashes = set()
            if os.path.isdir(label_dir):
                for f in os.listdir(label_dir):
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        local_hashes.add(_sha256_file(os.path.join(label_dir, f)))
            for entry in entries:
                if entry["sha256"] in local_hashes:
                    continue  # já temos esse conteúdo localmente, sob algum nome
                fname = entry["filename"]
                r = requests.get(f"{server_url}/imgs/download/{label}/{fname}", headers=headers(token), timeout=30)
                r.raise_for_status()
                dest = os.path.join(label_dir, fname)
                if os.path.exists(dest):
                    # nome já usado localmente por OUTRO conteúdo -- renomeia
                    stem, ext = os.path.splitext(fname)
                    n = 1
                    while os.path.exists(dest):
                        dest = os.path.join(label_dir, f"{stem}_remote{n}{ext}")
                        n += 1
                with open(dest, "wb") as f:
                    f.write(r.content)
                downloaded += 1
        print(f"  imgs/: {downloaded} foto(s) nova(s) baixada(s)")


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def sync_checkpoints(server_url, token, checkpoints_dir, push=True, pull=True):
    os.makedirs(checkpoints_dir, exist_ok=True)

    for prefix in CHECKPOINT_PREFIXES:
        meta_path = os.path.join(checkpoints_dir, f"{prefix}_latest.meta.json")
        local_meta = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                local_meta = json.load(f)

        remote_resp = requests.get(f"{server_url}/checkpoints/{prefix}/meta", headers=headers(token), timeout=30)
        remote_resp.raise_for_status()
        remote_meta = remote_resp.json()

        local_acc = local_meta.get("val_accuracy") if local_meta else None
        remote_acc = remote_meta.get("val_accuracy") if remote_meta.get("trained") else None

        if push and local_meta is not None and (remote_acc is None or (local_acc is not None and local_acc >= remote_acc)):
            model_path = os.path.join(checkpoints_dir, f"{prefix}_latest.pt")
            labels_path = os.path.join(checkpoints_dir, f"{prefix}_latest_labels.json")
            if os.path.exists(model_path) and os.path.exists(labels_path):
                with open(model_path, "rb") as mf, open(labels_path, "rb") as lf:
                    r = requests.post(
                        f"{server_url}/checkpoints/{prefix}/upload", headers=headers(token),
                        data={"meta": json.dumps(local_meta)},
                        files={"model": (f"{prefix}_latest.pt", mf), "labels": (f"{prefix}_latest_labels.json", lf)},
                        timeout=120,
                    )
                r.raise_for_status()
                result = r.json()
                print(f"  {prefix}: {'enviado e promovido' if result.get('promoted') else 'enviado mas não promovido (servidor já tinha melhor)'}")

        if pull and remote_meta.get("trained") and (local_acc is None or (remote_acc is not None and remote_acc > local_acc)):
            for kind, fname in [("model", f"{prefix}_latest.pt"), ("labels", f"{prefix}_latest_labels.json"), ("meta", f"{prefix}_latest.meta.json")]:
                r = requests.get(f"{server_url}/checkpoints/{prefix}/download/{kind}", headers=headers(token), timeout=120)
                r.raise_for_status()
                with open(os.path.join(checkpoints_dir, fname), "wb") as f:
                    f.write(r.content)
            print(f"  {prefix}: baixado do servidor (val_accuracy={remote_acc:.3f} > local)")


def main():
    parser = argparse.ArgumentParser(description="Sincroniza dados LIBRAS entre este PC e o servidor.")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--checkpoints-dir", default=os.path.join(BASE_DIR, "nn", "checkpoints"))
    parser.add_argument("--only", choices=["json", "imgs", "checkpoints"], default=None,
                         help="Sincroniza só uma categoria (padrão: todas)")
    parser.add_argument("--push-only", action="store_true")
    parser.add_argument("--pull-only", action="store_true")
    args = parser.parse_args()

    push = not args.pull_only
    pull = not args.push_only

    server_url, api_token = load_config(args)
    print(f"Servidor: {server_url}\n")

    try:
        r = requests.get(f"{server_url}/health", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"ERRO: não foi possível conectar ao servidor ({e}).")
        sys.exit(1)

    if args.only in (None, "json"):
        print("Sincronizando dados brutos (JSON)...")
        for name in RAW_JSON_NAMES:
            sync_json(server_url, api_token, name, push=push, pull=pull)

    if args.only in (None, "imgs"):
        print("\nSincronizando imgs/...")
        sync_imgs(server_url, api_token, push=push, pull=pull)

    if args.only in (None, "checkpoints"):
        print("\nSincronizando checkpoints...")
        sync_checkpoints(server_url, api_token, args.checkpoints_dir, push=push, pull=pull)

    print("\nSincronização concluída.")


if __name__ == "__main__":
    main()
