"""
libras_status.py - barramento de status do projeto LIBRAS
-----------------------------------------------------------
Módulo pequeno, sem dependências, usado por 4 peças:

    imgs_auditor_qwen.py  -> chama audit_start / audit_item / audit_finish
    train.py (seu)        -> chama publish_training(...)   [1 linha, ver abaixo]
    dashboard_server.py   -> chama collect_snapshot() e serve o painel web
    status_overlay.py     -> chama collect_snapshot() e desenha no translator

Tudo conversa por UM arquivo: libras_status.json (escrita atômica, então
ninguém lê arquivo pela metade). Env var LIBRAS_STATUS_FILE muda o caminho.

Hook de treino (cole no seu train.py, dentro do loop de épocas):
    import libras_status as st
    st.publish_training("estatica", epoch=ep, total_epochs=EPOCHS,
                        loss=train_loss, acc=train_acc, val_acc=val_acc)
    ...e no fim do treino:  st.publish_training("estatica", done=True)
(kind: "estatica" ou "movimento")

"Força" de um alvo (0-100), mostrada no painel e no translator:
    50% tamanho do pool de amostras (data_raw/commands_raw/movement_raw,
        chega a 100% com LIBRAS_POOL_TARGET amostras, padrão 40)
  + 50% taxa de fotos 'ok' na auditoria (só entra se o alvo foi auditado;
        senão a força é só o pool).
É um indicador de reforço da BASE, não a acurácia da rede.
"""

import os
import json
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.environ.get("LIBRAS_STATUS_FILE", os.path.join(BASE_DIR, "libras_status.json"))
POOL_TARGET = float(os.environ.get("LIBRAS_POOL_TARGET", "40"))
COMMANDS = ["BACKSPACE", "SPACE", "CLEAR"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CKPT_EXT = (".pt", ".pth", ".ckpt", ".pkl")

_lock = threading.Lock()
_audit = {}
_audit_last_flush = 0.0
HISTORY_CAP = 300
RECENT_CAP = 40


# ------------------------------------------------------------ arquivo
def read_status(path=None):
    path = path or STATUS_PATH
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            time.sleep(0.03)
    return {}


def _write(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    for _ in range(6):  # no Windows o replace falha se alguém estiver lendo
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.03)


def _mutate(fn, path=None):
    path = path or STATUS_PATH
    with _lock:
        state = read_status(path)
        fn(state)
        state["updated"] = time.time()
        try:
            _write(state, path)
        except OSError:
            pass  # painel é acessório: nunca derruba o script principal


# ------------------------------------------------------------ auditoria
def audit_start(total, model, mode, path=None):
    global _audit
    _audit = {"running": True, "total": total, "processed": 0,
              "counts": {"ok": 0, "revisar": 0, "rejeitar": 0}, "per_label": {},
              "current": "", "recent": [], "model": model, "mode": mode,
              "started": time.time(), "finished": None}
    _flush_audit(path)


def audit_item(rec, processed, path=None):
    global _audit_last_flush
    if not _audit:
        return
    a, fin = _audit, rec["final"]
    a["processed"] = processed
    a["counts"][fin] += 1
    pl = a["per_label"].setdefault(rec["rotulo"], {"ok": 0, "revisar": 0, "rejeitar": 0})
    pl[fin] += 1
    a["current"] = rec["arquivo"]
    q = rec.get("qwen")
    if fin != "ok" or q:
        q = q or {}
        a["recent"].append({"t": time.time(), "arquivo": rec["arquivo"], "rotulo": rec["rotulo"],
                            "final": fin, "qwen": q.get("veredito"), "conf": q.get("confianca"),
                            "sugestao": q.get("sugestao"), "motivo": str(q.get("raciocinio", ""))[:160]})
        a["recent"] = a["recent"][-RECENT_CAP:]
    if time.time() - _audit_last_flush > 0.3 or processed >= a["total"]:
        _flush_audit(path)


def audit_finish(stats=None, path=None):
    if not _audit:
        return
    _audit["running"] = False
    _audit["finished"] = time.time()
    _flush_audit(path)


def _flush_audit(path=None):
    global _audit_last_flush
    _audit_last_flush = time.time()
    snap = json.loads(json.dumps(_audit))
    _mutate(lambda s: s.__setitem__("audit", snap), path)


# ------------------------------------------------------------ treino
def publish_training(kind, epoch=None, total_epochs=None, loss=None, acc=None,
                     val_acc=None, done=False, path=None):
    def fn(state):
        t = state.setdefault("training", {"history": [], "last": {}, "running": False})
        if done:
            t["running"] = False
            t["last"]["finished"] = time.time()
            return
        t["running"] = True
        point = {"t": time.time(), "kind": kind, "epoch": epoch, "total": total_epochs,
                 "loss": loss, "acc": acc, "val_acc": val_acc}
        t["history"].append(point)
        t["history"] = t["history"][-HISTORY_CAP:]
        t["last"] = point
    _mutate(fn, path)


# ------------------------------------------------------------ snapshot
def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _depth(v):
    d = 0
    while isinstance(v, list) and v:
        v = v[0]
        d += 1
    return d


def _pool_counts(path, sample_depth):
    out = {}
    for name, v in _load(path).items():
        if not isinstance(v, list) or not v:
            out[name] = 0
        else:
            out[name] = 1 if _depth(v) == sample_depth else len(v)
    return out


def _img_counts(imgs_dir):
    out = {}
    if imgs_dir and os.path.isdir(imgs_dir):
        for d in os.listdir(imgs_dir):
            p = os.path.join(imgs_dir, d)
            if os.path.isdir(p) and not d.startswith("_"):
                n = sum(1 for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
                out["Ç" if d.upper() == "CA" else d.upper()] = n
    return out


def _checkpoints(ckpt_dir):
    out = []
    if ckpt_dir and os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.lower().endswith(CKPT_EXT):
                p = os.path.join(ckpt_dir, f)
                out.append({"nome": f, "mtime": os.path.getmtime(p), "kb": round(os.path.getsize(p) / 1024)})
    return sorted(out, key=lambda c: -c["mtime"])


def collect_snapshot(base_dir=None, imgs_dir=None, checkpoints_dir=None, status_path=None):
    base_dir = base_dir or BASE_DIR
    imgs_dir = imgs_dir or os.path.join(base_dir, "imgs")
    static = _pool_counts(os.path.join(base_dir, "data_raw.json"), 2)
    cmds = _pool_counts(os.path.join(base_dir, "commands_raw.json"), 2)
    mov = _pool_counts(os.path.join(base_dir, "movement_raw.json"), 3)
    imgs = _img_counts(imgs_dir)
    status = read_status(status_path)
    audit = status.get("audit") or {}
    per_label = audit.get("per_label", {})

    labels = sorted(set(static) | set(cmds) | set(mov) | set(imgs) | set(per_label))
    rows = []
    for l in labels:
        pool_s, pool_m = static.get(l, 0) + cmds.get(l, 0), mov.get(l, 0)
        n = pool_s + pool_m
        a = per_label.get(l, {"ok": 0, "revisar": 0, "rejeitar": 0})
        tot = a["ok"] + a["revisar"] + a["rejeitar"]
        pool_score = min(n / POOL_TARGET, 1.0)
        forca = 100 * (0.5 * pool_score + 0.5 * a["ok"] / tot) if tot else 100 * pool_score
        rows.append({"label": l,
                     "tipo": "comando" if l in COMMANDS else ("movimento" if pool_m and not pool_s else "letra"),
                     "fotos": imgs.get(l, 0), "pool_estatico": pool_s, "pool_mov": pool_m,
                     "audit": a, "forca": round(forca, 1)})

    ck_dir = checkpoints_dir
    return {"now": time.time(), "labels": rows, "audit": audit,
            "training": status.get("training") or {"history": [], "last": {}, "running": False},
            "checkpoints": _checkpoints(ck_dir),
            "totals": {"pool": sum(r["pool_estatico"] + r["pool_mov"] for r in rows),
                       "fotos": sum(r["fotos"] for r in rows),
                       "forca_media": round(sum(r["forca"] for r in rows) / len(rows), 1) if rows else 0.0},
            "pool_target": POOL_TARGET}