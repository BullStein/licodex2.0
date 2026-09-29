"""
status_overlay.py - painel de treino/reforço desenhado no translator
-----------------------------------------------------------------------
Lê o mesmo libras_status.json + pools que o dashboard_server.py e
desenha um painel semitransparente no canto direito do vídeo:
    - progresso da auditoria do Qwen (ok / revisar / rejeitar)
    - último ponto de treino (época, loss, acc) + mini-gráfico da loss
    - há quanto tempo o checkpoint da rede foi atualizado
    - FORÇA da letra que você está fazendo agora (pool + auditoria)

A coleta roda numa thread de fundo (a cada 2s), então ler os JSON
grandes nunca trava o vídeo. Tecla "p" liga/desliga o painel.
Se algo falhar, o painel some em silêncio -- o tradutor segue normal.
"""

import os
import time
import threading

import cv2

import libras_status as st


class StatusOverlay:
    def __init__(self, base_dir, nn_dir=None, status_file=None, refresh=2.0, visible=True):
        self.base_dir = base_dir
        self.ckpt = os.path.join(nn_dir, "checkpoints") if nn_dir else None
        self.status_file = status_file
        self.refresh = refresh
        self.visible = visible
        self.snap = None
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def toggle(self):
        self.visible = not self.visible

    def close(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                self.snap = st.collect_snapshot(self.base_dir, None, self.ckpt, self.status_file)
            except Exception:
                pass
            time.sleep(self.refresh)

    @staticmethod
    def _ago(t):
        s = max(0, time.time() - t)
        return f"{int(s)}s" if s < 90 else (f"{int(s / 60)}min" if s < 5400 else f"{int(s / 3600)}h")

    @staticmethod
    def _bar(frame, x, y, w, frac, color, h=8):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), -1)
        cv2.rectangle(frame, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)

    def draw(self, frame, letter=None, conf=0.0):
        if not self.visible or not self.snap:
            return
        s = self.snap
        H, W = frame.shape[:2]
        pw, ph = 310, 262
        x0, y0 = W - pw - 8, min(200, max(0, H - ph - 70))
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        F = cv2.FONT_HERSHEY_SIMPLEX
        x, y = x0 + 10, y0 + 20
        cv2.putText(frame, "TREINO / REFORCO  (p = esconde)", (x, y), F, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        # --- auditoria
        a = s.get("audit") or {}
        y += 22
        if a.get("total"):
            c = a["counts"]
            state = "..." if a.get("running") else "ok"
            cv2.putText(frame, f"Auditoria {a['processed']}/{a['total']} {state}", (x, y), F, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            self._bar(frame, x, y + 6, pw - 20, a["processed"] / a["total"], (255, 200, 80))
            y += 28
            cv2.putText(frame, f"ok {c['ok']}  revisar {c['revisar']}  rej {c['rejeitar']}", (x, y), F, 0.4,
                        (170, 220, 170), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Auditoria: sem dados", (x, y), F, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
            y += 18

        # --- treino
        t = s.get("training") or {}
        last, hist = t.get("last") or {}, t.get("history") or []
        y += 24
        if last:
            loss = f"{last['loss']:.3f}" if last.get("loss") is not None else "-"
            acc = f"{last['acc']:.2f}" if last.get("acc") is not None else "-"
            txt = f"Treino ep {last.get('epoch', '-')}/{last.get('total', '-')} loss {loss} acc {acc}"
            cv2.putText(frame, txt + (" *" if t.get("running") else ""), (x, y), F, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            vals = [p["loss"] for p in hist[-60:] if p.get("loss") is not None]
            if len(vals) >= 2:
                top, bot = max(vals), min(vals)
                span = (top - bot) or 1e-9
                gx, gy, gw, gh = x, y + 8, pw - 20, 26
                pts = [(gx + int(i * gw / (len(vals) - 1)), gy + gh - int((v - bot) / span * gh))
                       for i, v in enumerate(vals)]
                for p1, p2 in zip(pts, pts[1:]):
                    cv2.line(frame, p1, p2, (80, 80, 255), 1, cv2.LINE_AA)
            y += 42
        else:
            cv2.putText(frame, "Treino: sem dados (publish_training)", (x, y), F, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            y += 18

        ck = (s.get("checkpoints") or [None])[0]
        y += 6
        cv2.putText(frame, f"Rede atualizada ha {self._ago(ck['mtime'])}" if ck else "Rede: checkpoint nao achado (--nn-dir)",
                    (x, y), F, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # --- força da letra atual
        y += 26
        row = next((r for r in s.get("labels", []) if r["label"] == letter), None) if letter not in (None, "-", "?") else None
        if row:
            f = row["forca"]
            color = (60, 60, 240) if f < 40 else ((40, 180, 240) if f < 70 else (80, 200, 80))
            n = row["pool_estatico"] + row["pool_mov"]
            cv2.putText(frame, f"Letra {letter}: forca {f:.0f}%  ({n} amostras)", (x, y), F, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            self._bar(frame, x, y + 6, pw - 20, f / 100.0, color, h=10)
        else:
            cv2.putText(frame, "Faca uma letra pra ver a forca dela", (x, y), F, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
