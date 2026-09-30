"""
Comparador de fotos - LIBRAS
------------------------------
Compara CADA foto de imgs/<LETRA>/ com as demais fotos da mesma letra e
com as outras letras, e lista no terminal as que não batem:

    [TROCADA]       a foto está mais parecida com OUTRA letra do que com a
                    letra da pasta (provável foto na pasta errada)
    [OUTLIER]       a foto está longe do padrão da própria letra (margem de
                    erro grande: mão mal detectada, pose de transição, foto
                    borrada...)
    [REDE DISCORDA] (só com --nn-dir) a rede neural tem certeza de outra letra
    [SEM MAO]       o MediaPipe não achou mão na foto

Para cada problema mostra: a LETRA DA PASTA, o ARQUIVO, a letra com que a
foto parece e os números (distâncias e margem). No fim: resumo por letra e
as confusões mais comuns (ex.: "M -> N: 3 fotos"). Também grava um CSV
(comparador_relatorio.csv) com tudo, dá pra abrir no Excel/Power BI.

Como funciona: extrai os landmarks da mão (MediaPipe), normaliza igual aos
outros scripts e mede a distância média por landmark até o "padrão" (mediana)
de cada letra. A mediana da própria letra é calculada SEM a foto testada
(leave-one-out), então uma foto ruim não puxa o padrão a seu favor.
O limite de "margem grande" é por letra: mediana das distâncias + SIGMA
desvios robustos (ajuste com --sigma), nunca abaixo de --min-dist.

Não precisa de torch nem de Ollama. Com --nn-dir usa a rede como opinião
extra (se o torch estiver quebrado, segue sem ela).

Uso:
    python comparador_imgs.py
    python comparador_imgs.py --sigma 2.5            (mais rigoroso)
    python comparador_imgs.py --max-dist 0.25        (limite fixo pra todas)
    python comparador_imgs.py --only D,E,M           (só essas pastas)
    python comparador_imgs.py --nn-dir nn            (inclui a rede)

AÇÕES sobre as fotos problemáticas (escolha UMA; sempre teste antes com --dry-run):
    --quarantine   move pra imgs/_quarentena/<PASTA>/   (seguro, dá pra desfazer)
    --swap         move as TROCADA para a pasta da letra com que elas parecem
                   (renomeia pro próximo número livre, ex.: D_0043.jpg)
    --delete       APAGA de vez (pede pra você digitar EXCLUIR; sem volta)
    --undo         desfaz o último lote de quarentena/troca (não recupera excluídas)
  Extras: --dry-run (só mostra o que faria) | --yes (não pergunta)
          --status TROCADA,OUTLIER,REDE,SEM_MAO (quais problemas entram;
          padrão TROCADA,OUTLIER; o --swap só age em TROCADA)
  Tudo que for movido/apagado fica registrado em comparador_acoes.csv.

    python comparador_imgs.py --quarantine --dry-run
    python comparador_imgs.py --quarantine
    python comparador_imgs.py --swap
    python comparador_imgs.py --delete --status SEM_MAO
    python comparador_imgs.py --undo
"""

import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# torch ANTES de cv2/mediapipe (evita o WinError 1114 no Windows); só se a rede for usada
if "--nn-dir" in sys.argv:
    try:
        import torch  # noqa: F401
    except Exception as _e:
        print(f"Aviso: torch não carregou ({_e}). Seguindo sem a rede neural.")

import csv
import math
import shutil
import argparse
from collections import defaultdict, Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WRIST, MIDDLE_MCP = 0, 9
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
QUARANTINE_DIR = "_quarentena"
SEVERITY = {"TROCADA": 0, "REDE DISCORDA": 1, "OUTLIER": 2, "OK": 3}


# ------------------------------------------------------------ cores
def setup_color(enabled):
    if not enabled or not sys.stdout.isatty():
        return {k: "" for k in ("red", "yel", "mag", "gry", "grn", "bold", "end")}
    os.system("")  # ativa sequências ANSI no terminal do Windows 10+
    return {"red": "\033[91m", "yel": "\033[93m", "mag": "\033[95m", "gry": "\033[90m",
            "grn": "\033[92m", "bold": "\033[1m", "end": "\033[0m"}


# ------------------------------------------------------------ extração
def folder_to_label(folder):
    name = folder.strip().upper()
    return "Ç" if name == "CA" else name


def normalize_landmarks(landmarks):
    wrist, ref = landmarks[WRIST], landmarks[MIDDLE_MCP]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    scale = scale if scale > 1e-6 else 1e-6
    return np.array([[(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
                     for lm in landmarks], dtype=np.float64)


def extract_items(imgs_dir, only):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("Baixando hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    landmarker = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

    items, sem_mao = [], []
    folders = sorted(d for d in os.listdir(imgs_dir)
                     if os.path.isdir(os.path.join(imgs_dir, d)) and not d.startswith("_"))
    for folder in folders:
        label = folder_to_label(folder)
        if only and label not in only:
            continue
        fdir = os.path.join(imgs_dir, folder)
        files = sorted(f for f in os.listdir(fdir) if f.lower().endswith(VALID_EXT))
        print(f"  lendo {folder}/ ({len(files)} fotos)...", end="\r")
        for fname in files:
            path = os.path.join(fdir, fname)
            img = cv2.imread(path)
            res = None
            if img is not None:
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if img is None or not res.hand_landmarks:
                sem_mao.append({"label": label, "path": path})
                continue
            best = max(range(len(res.handedness)), key=lambda i: res.handedness[i][0].score)
            items.append({"path": path, "label": label, "pose": normalize_landmarks(res.hand_landmarks[best])})
    landmarker.close()
    print(" " * 50, end="\r")
    return items, sem_mao


# ------------------------------------------------------------ análise (numpy puro)
def pose_dist(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean())


def analyze(items, sigma=3.0, min_dist=0.08, max_dist=None, margin=0.02, nn_predict=None, nn_min_conf=0.7):
    by_label = defaultdict(list)
    for i, it in enumerate(items):
        by_label[it["label"]].append(i)
    medians = {l: np.median(np.stack([items[i]["pose"] for i in idx]), axis=0) for l, idx in by_label.items()}

    # distância de cada foto ao padrão da própria letra (leave-one-out) e às outras letras
    for l, idx in by_label.items():
        poses = np.stack([items[i]["pose"] for i in idx])
        for k, i in enumerate(idx):
            it = items[i]
            if len(idx) >= 3:
                own = np.median(np.delete(poses, k, axis=0), axis=0)
                it["d_own"] = pose_dist(it["pose"], own)
            else:
                it["d_own"] = None  # poucas fotos pra julgar
            others = {o: pose_dist(it["pose"], medians[o]) for o in medians if o != l}
            it["near"], it["d_near"] = min(others.items(), key=lambda x: x[1]) if others else (None, None)

    limits = {}
    for l, idx in by_label.items():
        d = np.array([items[i]["d_own"] for i in idx if items[i]["d_own"] is not None])
        if len(d) == 0:
            limits[l] = None
            continue
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) * 1.4826
        limits[l] = max_dist if max_dist else max(med + sigma * mad, min_dist)

    for it in items:
        it["rede"], it["rede_conf"] = None, None
        if nn_predict is not None:
            it["rede"], it["rede_conf"] = nn_predict(it["pose"])
        lim, d_own = limits[it["label"]], it["d_own"]
        it["limite"] = lim
        it["margem"] = (d_own - it["d_near"]) if (d_own is not None and it["d_near"] is not None) else None
        if d_own is None:
            it["status"] = "OK"
        elif it["d_near"] is not None and it["d_near"] + margin < d_own:
            it["status"] = "TROCADA"
        elif lim is not None and d_own > lim:
            it["status"] = "OUTLIER"
        elif it["rede"] and it["rede"] != it["label"] and (it["rede_conf"] or 0) >= nn_min_conf:
            it["status"] = "REDE DISCORDA"
        else:
            it["status"] = "OK"
    return items, limits


# ------------------------------------------------------------ relatório
def rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def print_report(items, sem_mao, limits, C, verbose=False):
    problems = [it for it in items if it["status"] != "OK"]
    by_label = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)
    sem_by = defaultdict(list)
    for s in sem_mao:
        sem_by[s["label"]].append(s)
    colors = {"TROCADA": C["red"], "OUTLIER": C["yel"], "REDE DISCORDA": C["mag"]}

    print(f"\n{C['bold']}================ FOTOS QUE NÃO BATEM ================{C['end']}")
    if not problems and not sem_mao:
        print(f"{C['grn']}Nenhuma foto suspeita. Tudo bate com a letra da pasta.{C['end']}")
    for label in sorted(set(by_label) | set(sem_by)):
        probs = sorted((it for it in by_label.get(label, []) if it["status"] != "OK"),
                       key=lambda it: (SEVERITY[it["status"]], -(it["d_own"] or 0)))
        sems = sem_by.get(label, [])
        if not probs and not sems and not verbose:
            continue
        total = len(by_label.get(label, [])) + len(sems)
        lim = limits.get(label)
        lim_txt = f" | limite de erro desta letra: {lim:.3f}" if lim else ""
        print(f"\n{C['bold']}PASTA '{label}'{C['end']} - {len(probs) + len(sems)} problema(s) em {total} fotos{lim_txt}")
        for it in probs:
            col = colors[it["status"]]
            if it["status"] == "TROCADA":
                msg = (f"parece a letra '{it['near']}'  (dist {it['near']}={it['d_near']:.3f} "
                       f"vs {label}={it['d_own']:.3f}, margem {it['margem']:+.3f})")
            elif it["status"] == "OUTLIER":
                msg = f"longe do padrão de '{label}'  (dist={it['d_own']:.3f} > limite {it['limite']:.3f}; mais próxima: '{it['near']}' {it['d_near']:.3f})"
            else:
                msg = f"rede acha '{it['rede']}' com conf {it['rede_conf']:.2f}  (dist própria {it['d_own']:.3f})"
            print(f"  {col}[{it['status']}]{C['end']} letra da pasta: {C['bold']}{label}{C['end']} | arquivo: {rel(it['path'])}\n"
                  f"      -> {msg}")
        for s in sems:
            print(f"  {C['gry']}[SEM MAO]{C['end']} letra da pasta: {C['bold']}{label}{C['end']} | arquivo: {rel(s['path'])}")
        if verbose:
            for it in by_label.get(label, []):
                if it["status"] == "OK":
                    d = f"{it['d_own']:.3f}" if it["d_own"] is not None else "-"
                    print(f"  {C['grn']}[OK]{C['end']} {rel(it['path'])} (dist {d})")

    # confusões
    conf = Counter((it["label"], it["near"]) for it in items if it["status"] == "TROCADA")
    if conf:
        print(f"\n{C['bold']}Confusões mais comuns (pasta -> parece):{C['end']}")
        for (a, b), n in conf.most_common(10):
            print(f"  {a} -> {b}: {n} foto(s)")

    # resumo
    print(f"\n{C['bold']}Resumo por letra{C['end']}   (fotos | ok | trocada | outlier | rede | sem mão | dist. média)")
    for label in sorted(set(by_label) | set(sem_by)):
        L = by_label.get(label, [])
        c = Counter(it["status"] for it in L)
        ds = [it["d_own"] for it in L if it["d_own"] is not None]
        dm = f"{np.mean(ds):.3f}" if ds else "-"
        bad = c["TROCADA"] + c["OUTLIER"] + c["REDE DISCORDA"] + len(sem_by.get(label, []))
        flag = C["red"] if bad >= max(3, 0.2 * (len(L) + len(sem_by.get(label, [])))) else ""
        print(f"  {flag}{label:>3}: {len(L) + len(sem_by.get(label, [])):3d} | {c['OK']:3d} | {c['TROCADA']:3d} | "
              f"{c['OUTLIER']:3d} | {c['REDE DISCORDA']:3d} | {len(sem_by.get(label, [])):3d} | {dm}{C['end']}")
    tot = Counter(it["status"] for it in items)
    print(f"\nTotal: {len(items)} fotos com mão | OK {tot['OK']} | TROCADA {tot['TROCADA']} | "
          f"OUTLIER {tot['OUTLIER']} | REDE DISCORDA {tot['REDE DISCORDA']} | SEM MÃO {len(sem_mao)}")


def write_csv(path, items, sem_mao):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["letra_da_pasta", "arquivo", "status", "parece_letra", "dist_propria", "dist_parece",
                    "margem", "limite_da_letra", "rede_sugestao", "rede_conf"])
        fmt = lambda v: "" if v is None else f"{v:.4f}".replace(".", ",")
        for it in sorted(items, key=lambda x: (x["label"], SEVERITY[x["status"]], x["path"])):
            w.writerow([it["label"], rel(it["path"]), it["status"], it["near"] or "", fmt(it["d_own"]),
                        fmt(it["d_near"]), fmt(it["margem"]), fmt(it["limite"]), it["rede"] or "", fmt(it["rede_conf"])])
        for s in sem_mao:
            w.writerow([s["label"], rel(s["path"]), "SEM MAO", "", "", "", "", "", "", ""])


# ------------------------------------------------------------ ações
STATUS_ALIASES = {"TROCADA": "TROCADA", "OUTLIER": "OUTLIER", "REDE": "REDE DISCORDA",
                  "REDE_DISCORDA": "REDE DISCORDA", "SEM_MAO": "SEM MAO", "SEMMAO": "SEM MAO"}


def folder_for_label(imgs_dir, label):
    name = "CA" if label == "Ç" else label
    for d in os.listdir(imgs_dir):
        if os.path.isdir(os.path.join(imgs_dir, d)) and d.upper() == name.upper():
            return os.path.join(imgs_dir, d)
    return os.path.join(imgs_dir, name)


def next_free_name(folder, label, ext):
    os.makedirs(folder, exist_ok=True)
    prefix = "CA" if label == "Ç" else label
    n = sum(1 for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)) + 1
    while True:
        path = os.path.join(folder, f"{prefix}_{n:04d}{ext}")
        if not os.path.exists(path):
            return path
        n += 1


def free_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{base}_dup{k}{ext}"):
        k += 1
    return f"{base}_dup{k}{ext}"


def plan_actions(items, sem_mao, acao, statuses, imgs_dir):
    """Devolve lista de (origem, destino|None, letra_da_pasta, letra_destino|None)."""
    plan = []
    if acao == "trocar":
        for it in items:
            if it["status"] == "TROCADA" and it["near"]:
                ext = os.path.splitext(it["path"])[1].lower() or ".jpg"
                dest = next_free_name(folder_for_label(imgs_dir, it["near"]), it["near"], ext)
                plan.append((it["path"], dest, it["label"], it["near"]))
        return plan
    cands = [(it["path"], it["label"]) for it in items if it["status"] in statuses]
    if "SEM MAO" in statuses:
        cands += [(sm["path"], sm["label"]) for sm in sem_mao]
    for path, label in cands:
        if acao == "quarentena":
            dst_dir = os.path.join(imgs_dir, QUARANTINE_DIR, os.path.basename(os.path.dirname(path)))
            plan.append((path, free_path(os.path.join(dst_dir, os.path.basename(path))), label, None))
        else:
            plan.append((path, None, label, None))
    return plan


def run_actions(plan, acao, log_path, dry_run=False, yes=False):
    verbo = {"quarentena": "QUARENTENA", "trocar": "TROCAR DE PASTA", "excluir": "EXCLUIR"}[acao]
    if not plan:
        print("Nenhuma foto se encaixa nessa ação.")
        return
    print(f"\n=== {verbo}: {len(plan)} foto(s) ===")
    for src, dst, lab, lab_dst in plan:
        if acao == "excluir":
            print(f"  apagar  [{lab}] {rel(src)}")
        else:
            extra = f" (letra {lab} -> {lab_dst})" if lab_dst else ""
            print(f"  mover   [{lab}] {rel(src)}  ->  {rel(dst)}{extra}")
    if dry_run:
        print("[DRY-RUN] nada foi alterado. Tire o --dry-run para executar.")
        return
    if not yes:
        try:
            if acao == "excluir":
                ok = input(f"\nIsso APAGA {len(plan)} foto(s) PARA SEMPRE. Digite EXCLUIR para confirmar: ").strip() == "EXCLUIR"
            else:
                ok = input(f"\nConfirmar {verbo.lower()} de {len(plan)} foto(s)? [s/N] ").strip().lower() in ("s", "sim", "y", "yes")
        except EOFError:
            ok = False
        if not ok:
            print("Cancelado. Nada foi alterado.")
            return

    import time as _t
    lote = _t.strftime("%Y%m%d-%H%M%S")
    new_log = not os.path.exists(log_path)
    done = failed = 0
    with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if new_log:
            w.writerow(["lote", "acao", "origem", "destino"])
        for src, dst, lab, lab_dst in plan:
            try:
                if acao == "excluir":
                    os.remove(src)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
                w.writerow([lote, acao, os.path.abspath(src), os.path.abspath(dst) if dst else ""])
                done += 1
            except OSError as e:
                failed += 1
                print(f"  ERRO em {rel(src)}: {e}")
    print(f"\n{done} foto(s) processada(s)" + (f", {failed} com erro" if failed else "") + f". Registro: {log_path}")
    if acao != "excluir":
        print("Para desfazer este lote:  python comparador_imgs.py --undo")
    print("Rode o comparador de novo para ver como ficou (as contagens por letra mudaram).")


def undo_last(log_path):
    if not os.path.exists(log_path):
        print("Nenhum registro de ações encontrado (comparador_acoes.csv).")
        return
    with open(log_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        print("Registro vazio.")
        return
    lote = max(r["lote"] for r in rows)
    batch = [r for r in rows if r["lote"] == lote]
    restored = skipped = gone = 0
    for r in reversed(batch):
        if r["acao"] == "excluir":
            gone += 1
            continue
        if os.path.exists(r["destino"]) and not os.path.exists(r["origem"]):
            os.makedirs(os.path.dirname(r["origem"]), exist_ok=True)
            shutil.move(r["destino"], r["origem"])
            restored += 1
        else:
            skipped += 1
    print(f"Lote {lote}: {restored} foto(s) devolvida(s) à pasta original"
          + (f", {skipped} ignorada(s) (arquivo não encontrado ou nome já ocupado)" if skipped else "")
          + (f", {gone} excluída(s) não dá(ão) pra recuperar" if gone else "") + ".")
    remaining = [r for r in rows if r["lote"] != lote]
    with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["lote", "acao", "origem", "destino"])
        for r in remaining:
            w.writerow([r["lote"], r["acao"], r["origem"], r["destino"]])


def main():
    ap = argparse.ArgumentParser(description="Mostra no terminal as fotos de imgs/ que não batem com a letra da pasta.")
    ap.add_argument("--imgs-dir", default=os.path.join(BASE_DIR, "imgs"))
    ap.add_argument("--only", default="", help="Só estas pastas, ex.: D,E,M")
    ap.add_argument("--sigma", type=float, default=3.0, help="Rigor do OUTLIER (menor = mais rigoroso, padrão 3.0)")
    ap.add_argument("--min-dist", type=float, default=0.08, help="Limite mínimo de erro por letra (padrão 0.08)")
    ap.add_argument("--max-dist", type=float, default=None, help="Limite FIXO de erro pra todas as letras")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="Quanto outra letra precisa estar mais perto pra marcar TROCADA (padrão 0.02)")
    ap.add_argument("--nn-dir", default=os.environ.get("LIBRAS_NN_DIR"), help="Inclui a rede neural como opinião extra")
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--nn-min-conf", type=float, default=0.7)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--quarantine", action="store_true", help="Move as fotos problemáticas pra imgs/_quarentena/")
    grp.add_argument("--swap", action="store_true", help="Move as TROCADA pra pasta da letra com que elas parecem")
    grp.add_argument("--delete", action="store_true", help="APAGA as fotos problemáticas (pede confirmação)")
    grp.add_argument("--undo", action="store_true", help="Desfaz o último lote de quarentena/troca")
    ap.add_argument("--status", default="TROCADA,OUTLIER",
                    help="Problemas que entram na ação: TROCADA,OUTLIER,REDE,SEM_MAO (padrão TROCADA,OUTLIER)")
    ap.add_argument("--dry-run", action="store_true", help="Só mostra o que a ação faria, sem alterar nada")
    ap.add_argument("--yes", action="store_true", help="Não pede confirmação")
    ap.add_argument("--log", default=os.path.join(BASE_DIR, "comparador_acoes.csv"))
    ap.add_argument("--csv", default=os.path.join(BASE_DIR, "comparador_relatorio.csv"))
    ap.add_argument("--verbose", action="store_true", help="Também lista as fotos OK")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    if args.undo:
        undo_last(args.log)
        return

    C = setup_color(not args.no_color)
    if not os.path.isdir(args.imgs_dir):
        sys.exit(f"Pasta '{args.imgs_dir}' não encontrada.")
    only = {folder_to_label(s) for s in args.only.split(",") if s.strip()}

    nn_predict = None
    if args.nn_dir:
        try:
            sys.path.insert(0, os.path.join(args.nn_dir, "inference"))
            sys.path.insert(0, os.path.join(args.nn_dir, "train"))
            from predictor import SignPredictor
            predictor = SignPredictor(args.checkpoints_dir or os.path.join(args.nn_dir, "checkpoints"))
            if predictor.static_ready:
                nn_predict = lambda pose: (lambda v: (v["sugestao"], v["confianca"]))(predictor.predict_static(pose))
            else:
                print("Aviso: rede sem checkpoint treinado; seguindo sem ela.")
        except Exception as e:
            print(f"Aviso: não foi possível carregar a rede ({type(e).__name__}: {e}); seguindo sem ela.")

    print(f"Lendo fotos de '{args.imgs_dir}'...")
    items, sem_mao = extract_items(args.imgs_dir, only)
    if not items:
        sys.exit("Nenhuma foto com mão detectada.")
    print(f"{len(items)} fotos com mão | {len(sem_mao)} sem mão")

    items, limits = analyze(items, args.sigma, args.min_dist, args.max_dist, args.margin, nn_predict, args.nn_min_conf)
    print_report(items, sem_mao, limits, C, args.verbose)
    write_csv(args.csv, items, sem_mao)
    print(f"\nRelatório completo: {args.csv}")

    acao = "quarentena" if args.quarantine else "trocar" if args.swap else "excluir" if args.delete else None
    if acao:
        statuses = set()
        for tok in args.status.split(","):
            tok = tok.strip().upper()
            if tok:
                if tok not in STATUS_ALIASES:
                    sys.exit(f"--status inválido: '{tok}'. Use TROCADA, OUTLIER, REDE ou SEM_MAO.")
                statuses.add(STATUS_ALIASES[tok])
        plan = plan_actions(items, sem_mao, acao, statuses, args.imgs_dir)
        run_actions(plan, acao, args.log, dry_run=args.dry_run, yes=args.yes)
    else:
        print("Dica: --quarantine (quarentena) | --swap (mandar pra pasta certa) | --delete (apagar) "
              "| --dry-run (simular) | --undo (desfazer)")


if __name__ == "__main__":
    main()