"""
Daemon de Treino Contínuo - LIBRAS NN Platform
--------------------------------------------------
Fica observando data_raw.json / commands_raw.json / movement_raw.json.
Toda vez que qualquer um deles muda, espera 30s de SILÊNCIO (nenhuma
mudança nova) antes de disparar um ciclo de treino completo -- isso
evita retreinar a cada amostra individual que o capture_signatures.py
ou o auto_validador_ia.py salvam, já que várias amostras costumam ser
capturadas em sequência numa mesma sessão.

Reusa 100% a lógica de train.py (run_training_cycle) -- este arquivo só
cuida de QUANDO disparar, não de COMO treinar.

Expõe métricas em /metrics (porta configurável, padrão 9308) pro
Prometheus coletar:
    - libras_last_retrain_timestamp_seconds{model="static|movement"}
    - libras_train_val_accuracy{model="static|movement"}
    - libras_train_val_loss{model="static|movement"}
    - libras_train_num_samples{model="static|movement"}
    - libras_train_num_classes{model="static|movement"}
    - libras_train_promoted_total{model="static|movement"}  (contador)
    - libras_train_cycles_total                              (contador)
    - libras_train_in_progress (0/1 gauge)

Uso básico:
    python train_daemon.py
    python train_daemon.py --debounce-seconds 30 --metrics-port 9308
    python train_daemon.py --watch-dir /caminho/onde/estao/os/json

Pra rodar em background de verdade (Linux/Mac):
    nohup python train_daemon.py > train_daemon.log 2>&1 &
No Windows, use um Task Scheduler apontando pro script, ou rode numa
janela separada / via `pythonw`.
"""

import argparse
import os
import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from prometheus_client import start_http_server, Gauge, Counter

from train import TrainConfig, run_training_cycle

WATCHED_FILENAMES = {"data_raw.json", "commands_raw.json", "movement_raw.json"}


# --------------------------------------------------------------------------
# Métricas Prometheus
# --------------------------------------------------------------------------

METRIC_LAST_RETRAIN = Gauge(
    "libras_last_retrain_timestamp_seconds", "Timestamp do último retreino", ["model"]
)
METRIC_VAL_ACCURACY = Gauge(
    "libras_train_val_accuracy", "Acurácia de validação do último treino", ["model"]
)
METRIC_VAL_LOSS = Gauge(
    "libras_train_val_loss", "Loss de validação do último treino", ["model"]
)
METRIC_NUM_SAMPLES = Gauge(
    "libras_train_num_samples", "Número de amostras usadas no último treino", ["model"]
)
METRIC_NUM_CLASSES = Gauge(
    "libras_train_num_classes", "Número de classes no último treino", ["model"]
)
METRIC_PROMOTED_TOTAL = Counter(
    "libras_train_promoted_total", "Quantas vezes um checkpoint foi promovido pra produção", ["model"]
)
METRIC_CYCLES_TOTAL = Counter(
    "libras_train_cycles_total", "Quantos ciclos de treino completos rodaram"
)
METRIC_IN_PROGRESS = Gauge(
    "libras_train_in_progress", "1 se um ciclo de treino está rodando agora, 0 caso contrário"
)


def _update_metrics_from_results(results: dict) -> None:
    now = time.time()
    for kind, result in results.items():
        if result is None:
            continue
        METRIC_LAST_RETRAIN.labels(model=kind).set(now)
        METRIC_VAL_ACCURACY.labels(model=kind).set(result.val_accuracy)
        METRIC_VAL_LOSS.labels(model=kind).set(result.val_loss)
        METRIC_NUM_SAMPLES.labels(model=kind).set(result.num_samples)
        METRIC_NUM_CLASSES.labels(model=kind).set(result.num_classes)
        if result.promoted:
            METRIC_PROMOTED_TOTAL.labels(model=kind).inc()


# --------------------------------------------------------------------------
# Debounce: agrupa rajadas de eventos de arquivo num único retreino
# --------------------------------------------------------------------------

class DebouncedRetrainHandler(FileSystemEventHandler):
    """
    A cada evento relevante (create/modify em um dos raw JSONs), reagenda
    um timer de `debounce_seconds`. Só quando o timer chega ao fim SEM ser
    reagendado de novo (ou seja, `debounce_seconds` de silêncio) é que o
    ciclo de treino de fato dispara -- evita retreinar a cada amostra
    individual salva em sequência.
    """

    def __init__(self, on_settle, debounce_seconds: float):
        self.on_settle = on_settle
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer = None
        self._lock = threading.Lock()

    def _is_relevant(self, path: str) -> bool:
        return os.path.basename(path) in WATCHED_FILENAMES

    def _schedule(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.on_settle)
            self._timer.daemon = True
            self._timer.start()

    def on_created(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule()

    def on_modified(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule()


# --------------------------------------------------------------------------
# Loop principal
# --------------------------------------------------------------------------

class TrainDaemon:
    def __init__(
        self,
        watch_dir: str,
        checkpoints_dir: str,
        debounce_seconds: float,
        cfg: TrainConfig,
    ):
        self.watch_dir = watch_dir
        self.checkpoints_dir = checkpoints_dir
        self.debounce_seconds = debounce_seconds
        self.cfg = cfg
        self._training_lock = threading.Lock()

    def _paths(self):
        return (
            os.path.join(self.watch_dir, "data_raw.json"),
            os.path.join(self.watch_dir, "commands_raw.json"),
            os.path.join(self.watch_dir, "movement_raw.json"),
        )

    def _run_cycle(self):
        # se um ciclo já está rodando (ex.: retreino demorado + nova mudança
        # chegou no meio), simplesmente ignora -- o próximo debounce que
        # settle depois vai capturar os dados mais recentes de qualquer forma.
        if not self._training_lock.acquire(blocking=False):
            print("[daemon] ciclo de treino já em andamento, ignorando novo disparo.")
            return
        try:
            METRIC_IN_PROGRESS.set(1)
            print(f"\n[daemon] {time.strftime('%Y-%m-%d %H:%M:%S')} -- "
                  f"{self.debounce_seconds:.0f}s de silêncio detectado, iniciando ciclo de treino...")
            data_raw, commands_raw, movement_raw = self._paths()
            results = run_training_cycle(
                data_raw, commands_raw, movement_raw, self.checkpoints_dir, self.cfg,
            )
            _update_metrics_from_results(results)
            METRIC_CYCLES_TOTAL.inc()
            print("[daemon] ciclo de treino concluído.\n")
        except Exception as e:
            # nunca deixa uma exceção matar o daemon -- só loga e segue
            # observando; a próxima mudança nos JSONs tenta de novo.
            print(f"[daemon] ERRO durante o ciclo de treino (daemon continua rodando): {e}")
        finally:
            METRIC_IN_PROGRESS.set(0)
            self._training_lock.release()

    def run_forever(self):
        os.makedirs(self.watch_dir, exist_ok=True)
        handler = DebouncedRetrainHandler(self._run_cycle, self.debounce_seconds)
        observer = Observer()
        observer.schedule(handler, self.watch_dir, recursive=False)
        observer.start()
        print(f"[daemon] observando '{self.watch_dir}' "
              f"(arquivos: {sorted(WATCHED_FILENAMES)}) -- debounce de {self.debounce_seconds:.0f}s")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[daemon] encerrando (Ctrl+C).")
        finally:
            observer.stop()
            observer.join()


def main():
    parser = argparse.ArgumentParser(description="Daemon de treino contínuo da LIBRAS NN Platform.")
    parser.add_argument("--watch-dir", default=".",
                         help="Pasta onde estão data_raw.json/commands_raw.json/movement_raw.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--debounce-seconds", type=float, default=30.0)
    parser.add_argument("--metrics-port", type=int, default=9308)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        val_ratio=args.val_ratio, patience=args.patience,
    )

    start_http_server(args.metrics_port)
    print(f"[daemon] métricas Prometheus em http://localhost:{args.metrics_port}/metrics")

    daemon = TrainDaemon(args.watch_dir, args.checkpoints_dir, args.debounce_seconds, cfg)
    daemon.run_forever()


if __name__ == "__main__":
    main()
