"""
patch_translator_nn.py - liga o painel de treino/reforço ao translator_nn.py
-------------------------------------------------------------------------------
Faz 4 edições pequenas no SEU translator_nn.py (cria translator_nn.py.bak antes):
    1. flags --no-panel e --status-file
    2. cria o StatusOverlay antes do loop principal
    3. desenha o painel antes do cv2.imshow
    4. tecla "p" liga/desliga o painel
Se o arquivo mudou e alguma âncora não bate, NADA é alterado.
Pode rodar mais de uma vez (detecta que já foi aplicado).

    python patch_translator_nn.py                    (usa ./translator_nn.py)
    python patch_translator_nn.py caminho/para/translator_nn.py
"""

import os
import sys
import shutil

path = sys.argv[1] if len(sys.argv) > 1 else "translator_nn.py"
src = open(path, encoding="utf-8").read()

if "StatusOverlay" in src:
    sys.exit("Já aplicado (StatusOverlay já existe no arquivo). Nada a fazer.")

EDITS = [
    (
        "    args = parser.parse_args()\n",
        '    parser.add_argument("--no-panel", action="store_true",\n'
        '                         help="Não desenha o painel de treino/reforço (tecla p também liga/desliga).")\n'
        '    parser.add_argument("--status-file", default=None,\n'
        '                         help="Caminho do libras_status.json (padrão: o do libras_status.py).")\n'
        "    args = parser.parse_args()\n",
    ),
    (
        "    letter_buffer = deque(maxlen=8)\n",
        "    panel = None\n"
        "    if not args.no_panel:\n"
        "        try:\n"
        "            from status_overlay import StatusOverlay\n"
        "            panel = StatusOverlay(os.path.dirname(os.path.abspath(__file__)),\n"
        "                                  nn_dir=args.nn_dir, status_file=args.status_file)\n"
        "        except ImportError:\n"
        '            print("AVISO: status_overlay.py/libras_status.py não encontrados; painel desligado.")\n'
        "\n"
        "    letter_buffer = deque(maxlen=8)\n",
    ),
    (
        "        cv2.imshow(WINDOW_NAME, frame)\n",
        "        if panel is not None:\n"
        "            panel.draw(frame, letter, last_letter_confidence)\n"
        "\n"
        "        cv2.imshow(WINDOW_NAME, frame)\n",
    ),
    (
        '        elif key == ord("9"):\n            letter_history = ""\n',
        '        elif key == ord("9"):\n            letter_history = ""\n'
        '        elif key == ord("p") and panel is not None:\n            panel.toggle()\n',
    ),
]

for old, _ in EDITS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"Âncora não bateu ({n} ocorrências) -- arquivo diferente do esperado, nada foi alterado:\n{old!r}")

for old, new in EDITS:
    src = src.replace(old, new, 1)

shutil.copyfile(path, path + ".bak")
open(path, "w", encoding="utf-8").write(src)
print(f"OK: {path} atualizado (backup em {os.path.basename(path)}.bak).")
print("Rode:  python translator_nn.py --nn-dir nn      | tecla 'p' liga/desliga o painel")
