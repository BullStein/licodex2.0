"""
Auditor da biblioteca imgs/ com Qwen (Ollama) - LIBRAS
---------------------------------------------------------
Passa por TODAS as fotos de imgs/<ALVO>/ ANTES do batch_train_from_images.py
e separa o joio do trigo, pra reforçar a base de treino:

  1) MediaPipe extrai e normaliza os landmarks de cada foto (mesma
     normalização dos outros scripts).
  2) Primeira opinião (barata, sem LLM):
       - geométrica: distância média ao "centroide" de cada alvo
         (leave-one-out pro próprio alvo);
       - rede neural (opcional, --nn-dir): SignPredictor.predict_static.
  3) Se as opiniões concordam com a pasta -> "ok" (Qwen nem é chamado).
     Se discordam -> a foto é SUSPEITA e o Qwen dá a segunda opinião.
  4) Qwen roda em um de dois modos:
       --mode text   (padrão) qwen3:4b  -> julga só pelos NÚMEROS
                     (ranking, distâncias, palpite da rede), sem ver a foto.
       --mode vision qwen3-vl:4b        -> além dos números, VÊ a foto.
  5) Veredito final por foto: ok | revisar | rejeitar.
     Com --quarantine, as "rejeitar" são MOVIDAS (nunca apagadas) para
     imgs/_quarentena/<ALVO>/ -- o batch_train ignora essa pasta.

AVISO HONESTO: um modelo de 4B parâmetros não é especialista em
handshapes de LIBRAS. Trate o Qwen como desempate/2ª opinião, não como
verdade. Por segurança, só rejeita quando o Qwen diz "errada" com
confiança alta E a checagem geométrica/rede também discorda da pasta.
Sempre olhe o relatório (imgs_audit.json) antes de treinar.

Pré-requisitos:
    ollama pull qwen3:4b           (modo text)
    ollama pull qwen3-vl:4b        (modo vision)
    hand_landmarker.task na mesma pasta (baixa sozinho se faltar)

Modelo/host: por padrão usa ollama_config.get_active_settings() (o mesmo
do ai_judge.py). Sobrescreva com --model / --host.

Como usar:
    python imgs_auditor_qwen.py
    python imgs_auditor_qwen.py --model qwen3:4b
    python imgs_auditor_qwen.py --mode vision --model qwen3-vl:4b
    python imgs_auditor_qwen.py --nn-dir nn            (inclui a rede neural)
    python imgs_auditor_qwen.py --judge-all            (Qwen julga TODAS)
    python imgs_auditor_qwen.py --quarantine           (move as rejeitadas)
    depois: python batch_train_from_images.py --augment --dedupe 0.02
"""

import os
import sys
import json
import math
import base64
import shutil
import argparse
import urllib.request
import urllib.error
from collections import defaultdict

# torch ANTES de cv2/mediapipe: no Windows a ordem contrária costuma causar
# "WinError 1114 ... c10.dll". Se o torch estiver quebrado, o script segue sem a rede.
try:
    import torch  # noqa: F401
except Exception as _torch_err:
    torch = None
    print(f"Aviso: torch não carregou ({_torch_err}). A rede neural fica desligada; "
          "seguindo só com checagem geométrica + Qwen.")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import libras_status as st  # alimenta o painel ao vivo (dashboard_server.py / translator)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WRIST, MIDDLE_MCP = 0, 9
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
QUARANTINE_DIR = "_quarentena"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def normalize_landmarks(landmarks):
    wrist, ref = landmarks[WRIST], landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array([[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
                     for lm in landmarks], dtype=np.float64)


def folder_to_label(folder):
    name = folder.strip().upper()
    return "Ç" if name == "CA" else name


def pose_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


# ---------------------------------------------------------------- Ollama
def get_ollama_settings(args):
    model, host = None, None
    try:
        from ollama_config import get_active_settings
        model, host, _ = get_active_settings(args.profile)
    except Exception:
        pass
    model = args.model or model or ("qwen3-vl:4b" if args.mode == "vision" else "qwen3:4b")
    host = args.host or host or "http://localhost:11434"
    return model, host


def call_qwen(prompt, model, host, timeout, image_path=None):
    msg = {"role": "user", "content": prompt}
    if image_path:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        if max(h, w) > 768:  # reduz pra ficar rápido e caber no contexto
            s = 768 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", img)
        msg["images"] = [base64.b64encode(buf.tobytes()).decode("ascii")]
    payload = {"model": model, "messages": [msg], "stream": False, "format": "json",
               "think": False, "options": {"temperature": 0.1}}
    req = urllib.request.Request(host.rstrip("/") + "/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")


def parse_json_loose(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        s, e = (raw or "").find("{"), (raw or "").rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


def build_prompt(label, geo_rank, nn, vision):
    ranking = "\n".join(f"{i + 1}. {n} (distância={d:.4f})" for i, (n, d) in enumerate(geo_rank[:5]))
    nn_txt = (f'A rede neural previu "{nn[0]}" com confiança {nn[1]:.2f}.' if nn
              else "Não há rede neural disponível.")
    olhar = ("Você também recebe a FOTO da mão. Confira se o formato da mão parece compatível com a letra/comando "
             "de LIBRAS indicado, mas seja cauteloso: só diga 'errada' se estiver claramente diferente.\n"
             if vision else "Você NÃO está vendo a foto; julgue só pelos números, sem inventar detalhes visuais.\n")
    return f"""Você audita um banco de fotos para treinar um reconhecedor de LIBRAS por landmarks de mão (MediaPipe).
Esta foto está na pasta rotulada "{label}".
Comparando os landmarks normalizados com o centroide de cada alvo (distância MENOR = mais parecido):
{ranking}
{nn_txt}
{olhar}
Avalie se o rótulo "{label}" desta foto é confiável. Se o rótulo não é o 1º do ranking e a distância dele é bem maior, é sinal de foto mal rotulada, mão mal detectada ou pose de transição.
Responda APENAS com JSON: {{"veredito": "correta" | "duvidosa" | "errada", "confianca": <0.0 a 1.0>, "sugestao": "<alvo mais provável>", "raciocinio": "<uma frase curta>"}}"""


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description="Audita imgs/ com MediaPipe + rede + Qwen (Ollama).")
    ap.add_argument("--imgs-dir", default=os.path.join(BASE_DIR, "imgs"))
    ap.add_argument("--mode", choices=["text", "vision"], default="text")
    ap.add_argument("--model", default=None, help="Ex.: qwen3:4b ou qwen3-vl:4b")
    ap.add_argument("--host", default=None)
    ap.add_argument("--profile", default=None, help="Perfil do ollama_config.py")
    ap.add_argument("--nn-dir", default=os.environ.get("LIBRAS_NN_DIR"))
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--nn-min-conf", type=float, default=0.6)
    ap.add_argument("--reject-conf", type=float, default=0.75,
                    help="Confiança mínima do Qwen pra rejeitar (padrão 0.75)")
    ap.add_argument("--judge-all", action="store_true", help="Qwen julga todas, não só as suspeitas")
    ap.add_argument("--quarantine", action="store_true", help="Move as rejeitadas pra imgs/_quarentena/")
    ap.add_argument("--skip-labels", default="", help="Alvos a ignorar, ex.: H,J,K,X,Z")
    ap.add_argument("--limit", type=int, default=0, help="Máx. de fotos (0 = todas), útil pra testar")
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("LIBRAS_OLLAMA_TIMEOUT", "90")))
    ap.add_argument("--report", default=os.path.join(BASE_DIR, "imgs_audit.json"))
    args = ap.parse_args()

    if not os.path.isdir(args.imgs_dir):
        sys.exit(f"Pasta '{args.imgs_dir}' não encontrada.")

    model, host = get_ollama_settings(args)
    print(f"Qwen -> modelo '{model}' | host {host} | modo {args.mode}")

    predictor = None
    if args.nn_dir:
        sys.path.insert(0, os.path.join(args.nn_dir, "inference"))
        sys.path.insert(0, os.path.join(args.nn_dir, "train"))
        try:
            from predictor import SignPredictor
            predictor = SignPredictor(args.checkpoints_dir or os.path.join(args.nn_dir, "checkpoints"))
            if not predictor.static_ready:
                print("Aviso: rede sem checkpoint treinado, seguindo só com a checagem geométrica.")
                predictor = None
        except Exception as e:
            print(f"Aviso: não foi possível carregar a rede ({type(e).__name__}: {e}). "
                  "Seguindo só com checagem geométrica + Qwen.")
            predictor = None

    skip = {folder_to_label(s) for s in args.skip_labels.split(",") if s.strip()}

    # cache pra poder interromper e continuar
    cache_path = args.report + ".cache"
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    ensure_model()
    landmarker = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

    # ---- 1) extrai poses
    items = []  # dict(path, folder, label, pose)
    sem_mao = []
    for folder in sorted(d for d in os.listdir(args.imgs_dir)
                         if os.path.isdir(os.path.join(args.imgs_dir, d)) and not d.startswith("_")):
        label = folder_to_label(folder)
        if label in skip:
            continue
        fdir = os.path.join(args.imgs_dir, folder)
        for fname in sorted(f for f in os.listdir(fdir) if f.lower().endswith(VALID_EXT)):
            path = os.path.join(fdir, fname)
            img = cv2.imread(path)
            if img is None:
                sem_mao.append(path)
                continue
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.hand_landmarks:
                sem_mao.append(path)
                continue
            best = max(range(len(res.handedness)), key=lambda i: res.handedness[i][0].score)
            items.append({"path": path, "folder": folder, "label": label,
                          "pose": normalize_landmarks(res.hand_landmarks[best])})
            if args.limit and len(items) >= args.limit:
                break
        if args.limit and len(items) >= args.limit:
            break
    landmarker.close()
    print(f"{len(items)} fotos com mão detectada | {len(sem_mao)} sem mão (ignoradas)")
    if not items:
        return
    st.audit_start(len(items), model, args.mode)

    # ---- 2) centroides por alvo
    by_label = defaultdict(list)
    for i, it in enumerate(items):
        by_label[it["label"]].append(i)
    sums = {l: np.sum([items[i]["pose"] for i in idx], axis=0) for l, idx in by_label.items()}

    def centroid_without(label, pose):
        n = len(by_label[label])
        if n <= 1:
            return sums[label] / max(n, 1)
        return (sums[label] - pose) / (n - 1)  # leave-one-out

    results = []
    stats = defaultdict(int)
    qwen_offline = False

    for k, it in enumerate(items, 1):
        pose, label = it["pose"], it["label"]
        geo = sorted(((l, pose_dist(pose, centroid_without(l, pose) if l == label else sums[l] / len(by_label[l])))
                      for l in by_label), key=lambda x: x[1])
        geo_ok = geo[0][0] == label

        nn = None
        if predictor is not None:
            v = predictor.predict_static(pose)
            nn = (v["sugestao"], v["confianca"])
        nn_ok = nn is None or nn[1] < args.nn_min_conf or nn[0] == label

        suspeita = not (geo_ok and nn_ok)
        rec = {"arquivo": os.path.relpath(it["path"], args.imgs_dir), "rotulo": label,
               "geo_top3": [(n, round(d, 4)) for n, d in geo[:3]],
               "rede": None if nn is None else {"sugestao": nn[0], "confianca": round(nn[1], 3)},
               "suspeita": suspeita, "qwen": None}

        if suspeita or args.judge_all:
            key = f"{it['path']}|{os.path.getmtime(it['path'])}|{model}|{args.mode}"
            if key in cache:
                rec["qwen"] = cache[key]
            elif not qwen_offline:
                try:
                    raw = call_qwen(build_prompt(label, geo, nn, args.mode == "vision"), model, host,
                                    args.timeout, it["path"] if args.mode == "vision" else None)
                    p = parse_json_loose(raw) or {}
                    try:
                        conf = max(0.0, min(1.0, float(p.get("confianca", 0.5))))
                    except (TypeError, ValueError):
                        conf = 0.5
                    ver = p.get("veredito") if p.get("veredito") in ("correta", "duvidosa", "errada") else "duvidosa"
                    rec["qwen"] = {"veredito": ver, "confianca": conf,
                                   "sugestao": p.get("sugestao"), "raciocinio": p.get("raciocinio", "")}
                    cache[key] = rec["qwen"]
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    qwen_offline = True
                    print(f"\nOllama indisponível ({e}). Seguindo sem Qwen (suspeitas viram 'revisar').")

        # ---- veredito final
        q = rec["qwen"]
        if not suspeita and not (q and q["veredito"] == "errada" and q["confianca"] >= args.reject_conf and args.judge_all):
            final = "ok"
        elif suspeita and q and q["veredito"] == "errada" and q["confianca"] >= args.reject_conf:
            final = "rejeitar"
        elif suspeita and q and q["veredito"] == "correta" and q["confianca"] >= args.reject_conf:
            final = "ok"
        else:
            final = "revisar"
        rec["final"] = final
        stats[final] += 1
        results.append(rec)
        st.audit_item(rec, k)
        if final != "ok":
            extra = f" | Qwen: {q['veredito']} ({q['confianca']:.2f}) {q['raciocinio']}" if q else ""
            print(f"[{k}/{len(items)}] {final.upper():8} {rec['arquivo']} (geo top1={geo[0][0]}){extra}")
        elif k % 50 == 0:
            print(f"[{k}/{len(items)}] ...")

    # ---- quarentena
    if args.quarantine:
        moved = 0
        for rec in results:
            if rec["final"] == "rejeitar":
                src = os.path.join(args.imgs_dir, rec["arquivo"])
                dst_dir = os.path.join(args.imgs_dir, QUARANTINE_DIR, os.path.dirname(rec["arquivo"]))
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, os.path.join(dst_dir, os.path.basename(src)))
                moved += 1
        print(f"{moved} foto(s) movida(s) para imgs/{QUARANTINE_DIR}/ (nada foi apagado).")

    json.dump({"modelo": model, "modo": args.mode, "resumo": dict(stats),
               "sem_mao": [os.path.relpath(p, args.imgs_dir) for p in sem_mao], "fotos": results},
              open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    st.audit_finish(dict(stats))

    per_label = defaultdict(lambda: defaultdict(int))
    for r in results:
        per_label[r["rotulo"]][r["final"]] += 1
    print("\n--- Resumo por alvo (ok / revisar / rejeitar) ---")
    for l in sorted(per_label):
        c = per_label[l]
        print(f"  {l}: {c['ok']} / {c['revisar']} / {c['rejeitar']}")
    print(f"\nTotal: {dict(stats)} | relatório: {args.report}")
    print("Próximo passo: python batch_train_from_images.py --augment --dedupe 0.02")


if __name__ == "__main__":
    main()
