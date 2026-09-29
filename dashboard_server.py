"""
Painel ao vivo - LIBRAS (estilo Power BI, sem instalar nada)
---------------------------------------------------------------
Sobe um servidor local e abre um painel no navegador que atualiza a
cada 1s mostrando, em tempo real:
    - auditoria do imgs_auditor_qwen.py (progresso, ok/revisar/rejeitar,
      últimos vereditos do Qwen)
    - "força" e tamanho do pool de cada letra/comando/movimento, com o
      +N amostras ganhas desde que você abriu a página (o reforço acontecendo)
    - curvas de treino (loss/acc) se o train.py chamar publish_training()
    - checkpoints da rede (quando o último foi atualizado)

Só usa a biblioteca padrão do Python. Rode na pasta do projeto:
    python dashboard_server.py
    python dashboard_server.py --nn-dir nn
    python dashboard_server.py --port 8765 --no-browser

Power BI de verdade (opcional): Obter dados > Web >
    http://localhost:8765/api/export.csv
e atualize/agende a atualização lá. (Deve funcionar no Power BI Desktop;
não testei.)
"""

import os
import csv
import io
import json
import argparse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import libras_status as st

HTML = r"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBRAS - Painel ao vivo</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--line:#262d38;--tx:#e6e9ee;--mut:#8b95a5;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui,Segoe UI,sans-serif;padding:18px}
h1{font-size:18px;margin:0 0 14px;display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--bad)}.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
.grid{display:grid;gap:12px}.kpis{grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:26px;font-weight:600;margin-top:4px}
.sub{color:var(--mut);font-size:12px;margin-top:4px}
.bar{height:8px;background:var(--line);border-radius:5px;overflow:hidden;margin-top:8px;display:flex}
.bar>i{display:block;height:100%}
.two{grid-template-columns:2fr 1fr}@media(max-width:900px){.two{grid-template-columns:1fr}}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.tile{background:#11161d;border:1px solid var(--line);border-radius:8px;padding:8px;transition:border-color .3s,box-shadow .3s}
.tile.flash{border-color:var(--acc);box-shadow:0 0 12px rgba(88,166,255,.5)}
.tile .l{font-size:20px;font-weight:700}.tile .t{font-size:10px;color:var(--mut);float:right}
.tile .n{font-size:11px;color:var(--mut);margin-top:4px}.plus{color:var(--ok)}
h2{font-size:13px;margin:0 0 10px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.feed{max-height:340px;overflow:auto;font-size:12px}.ev{padding:6px 0;border-bottom:1px solid var(--line)}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px}
.t-ok{background:rgba(63,185,80,.18);color:var(--ok)}.t-revisar{background:rgba(210,153,34,.18);color:var(--warn)}.t-rejeitar{background:rgba(248,81,73,.18);color:var(--bad)}
svg{width:100%;height:190px;background:#11161d;border-radius:8px}
.leg span{margin-right:14px;font-size:12px}.leg b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style></head><body>
<h1><span id="dot" class="dot"></span>LIBRAS - Painel ao vivo <span class="sub" id="upd"></span></h1>
<div class="grid kpis">
 <div class="card"><div class="k">Auditoria</div><div class="v" id="k_aud">-</div><div class="bar" id="b_aud"></div><div class="sub" id="s_aud"></div></div>
 <div class="card"><div class="k">Força média da base</div><div class="v" id="k_forca">-</div><div class="bar"><i id="b_forca" style="background:var(--acc)"></i></div><div class="sub" id="s_forca"></div></div>
 <div class="card"><div class="k">Amostras no pool</div><div class="v" id="k_pool">-</div><div class="sub" id="s_pool"></div></div>
 <div class="card"><div class="k">Último checkpoint</div><div class="v" id="k_ck">-</div><div class="sub" id="s_ck"></div></div>
</div>
<div class="grid two">
 <div class="card"><h2>Força por alvo <span class="sub">(pool + auditoria) - borda azul = ganhou amostra agora</span></h2><div class="tiles" id="tiles"></div></div>
 <div class="card"><h2>Qwen / auditoria - últimos eventos</h2><div class="feed" id="feed"><span class="sub">Rode o imgs_auditor_qwen.py</span></div></div>
</div>
<div class="grid" style="margin-top:12px"><div class="card"><h2>Treino <span id="tinfo" class="sub"></span></h2>
 <svg id="chart" viewBox="0 0 800 190" preserveAspectRatio="none"></svg>
 <div class="leg"><span><b style="background:var(--bad)"></b>loss</span><span><b style="background:var(--ok)"></b>acc</span><span><b style="background:var(--acc)"></b>val_acc</span></div>
</div></div>
<script>
const $=id=>document.getElementById(id);let first=null,lastPool={},flash={};
const col=f=>f<40?'var(--bad)':f<70?'var(--warn)':'var(--ok)';
const ago=t=>{const s=Math.max(0,Date.now()/1000-t);return s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+' min':Math.round(s/3600)+' h'};
function line(pts,color,max){if(pts.length<2)return'';const d=pts.map((p,i)=>`${(i/(pts.length-1))*800},${185-(p/max)*175}`).join(' ');return`<polyline fill="none" stroke="${color}" stroke-width="2" points="${d}"/>`}
function render(s){
 $('upd').textContent='atualizado '+new Date().toLocaleTimeString();
 const a=s.audit||{};
 if(a.total){const p=a.processed/a.total,c=a.counts;$('k_aud').textContent=a.processed+' / '+a.total+(a.running?' ...':' ✓');
  $('b_aud').innerHTML=['ok','revisar','rejeitar'].map(k=>`<i style="width:${(c[k]/a.total)*100}%;background:var(--${k=='ok'?'ok':k=='revisar'?'warn':'bad'})"></i>`).join('');
  $('s_aud').textContent=`ok ${c.ok} · revisar ${c.revisar} · rejeitar ${c.rejeitar} · ${a.model||''} (${a.mode||''})`}
 const T=s.totals;$('k_forca').textContent=T.forca_media+'%';$('b_forca').style.width=T.forca_media+'%';
 $('s_forca').textContent='meta do pool: '+s.pool_target+' amostras/alvo';
 $('k_pool').textContent=T.pool;$('s_pool').textContent=T.fotos+' fotos em imgs/';
 const ck=s.checkpoints[0];$('k_ck').textContent=ck?ago(ck.mtime):'-';$('s_ck').textContent=ck?ck.nome:'nenhum checkpoint encontrado (use --nn-dir)';
 if(!first){first={};s.labels.forEach(r=>first[r.label]=r.pool_estatico+r.pool_mov)}
 $('tiles').innerHTML=s.labels.map(r=>{const n=r.pool_estatico+r.pool_mov;
  if(lastPool[r.label]!==undefined&&n>lastPool[r.label])flash[r.label]=Date.now();lastPool[r.label]=n;
  const d=n-(first[r.label]??n),au=r.audit,tt=au.ok+au.revisar+au.rejeitar,f=Date.now()-(flash[r.label]||0)<6000;
  return`<div class="tile${f?' flash':''}"><span class="t">${r.tipo}</span><div class="l">${r.label}</div>
  <div class="bar"><i style="width:${r.forca}%;background:${col(r.forca)}"></i></div>
  <div class="n">${r.forca}% · ${n} amostras${d>0?` <span class="plus">+${d}</span>`:''}</div>
  <div class="n">${r.fotos} fotos${tt?` · ok ${Math.round(100*au.ok/tt)}%`:''}</div></div>`}).join('');
 if(a.recent&&a.recent.length)$('feed').innerHTML=a.recent.slice().reverse().map(e=>`<div class="ev"><span class="tag t-${e.final}">${e.final}</span><b>${e.rotulo}</b> ${e.arquivo.split(/[\\/]/).pop()}
  ${e.qwen?`<br><span class="sub">Qwen: ${e.qwen} (${(e.conf??0).toFixed(2)})${e.sugestao?' → '+e.sugestao:''} ${e.motivo||''}</span>`:''}</div>`).join('');
 const h=(s.training.history||[]).slice(-120),L=s.training.last||{};
 const loss=h.map(p=>p.loss).filter(x=>x!=null),acc=h.map(p=>p.acc).filter(x=>x!=null),va=h.map(p=>p.val_acc).filter(x=>x!=null);
 $('chart').innerHTML=line(loss,'#f85149',Math.max(...loss,1e-6))+line(acc,'#3fb950',1)+line(va,'#58a6ff',1);
 $('tinfo').textContent=h.length?`${L.kind||''} · época ${L.epoch??'-'}/${L.total??'-'} · loss ${L.loss?.toFixed?.(4)??'-'} · acc ${L.acc?.toFixed?.(3)??'-'}${s.training.running?' · treinando...':''}`:'sem dados - chame publish_training() no train.py';
}
async function tick(){try{render(await (await fetch('/api/status')).json());$('dot').className='dot on'}catch(e){$('dot').className='dot'}}
setInterval(tick,1000);tick();
</script></body></html>"""


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, HTML, "text/html; charset=utf-8")
            snap = None
            if path in ("/api/status", "/api/export.csv"):
                snap = st.collect_snapshot(cfg["base"], cfg["imgs"], cfg["ckpt"], cfg["status"])
            if path == "/api/status":
                return self._send(200, json.dumps(snap, ensure_ascii=False), "application/json; charset=utf-8")
            if path == "/api/export.csv":
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["alvo", "tipo", "fotos", "pool_estatico", "pool_mov", "ok", "revisar", "rejeitar", "forca"])
                for r in snap["labels"]:
                    a = r["audit"]
                    w.writerow([r["label"], r["tipo"], r["fotos"], r["pool_estatico"], r["pool_mov"],
                                a["ok"], a["revisar"], a["rejeitar"], r["forca"]])
                return self._send(200, "\ufeff" + buf.getvalue(), "text/csv; charset=utf-8")
            self._send(404, "não encontrado", "text/plain; charset=utf-8")
    return Handler


def main():
    ap = argparse.ArgumentParser(description="Painel ao vivo do projeto LIBRAS.")
    ap.add_argument("--base-dir", default=st.BASE_DIR, help="Pasta com data_raw.json etc.")
    ap.add_argument("--imgs-dir", default=None)
    ap.add_argument("--nn-dir", default=os.environ.get("LIBRAS_NN_DIR"))
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--status-file", default=st.STATUS_PATH)
    ap.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 pra abrir na rede local")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoints_dir or (os.path.join(args.nn_dir, "checkpoints") if args.nn_dir else None)
    cfg = {"base": args.base_dir, "imgs": args.imgs_dir or os.path.join(args.base_dir, "imgs"),
           "ckpt": ckpt, "status": args.status_file}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cfg))
    url = f"http://localhost:{args.port}"
    print(f"Painel em {url}  (Ctrl+C pra parar)\nStatus: {args.status_file}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
