"""
Métricas de Inferência - LIBRAS NN Platform
------------------------------------------------
Complementa as métricas de TREINO que o train_daemon.py já expõe
(libras_train_*) com métricas do lado da INFERÊNCIA em produção --
ou seja, do processo que roda a webcam de verdade usando o
inference/predictor.py.

Uso básico (dentro do translator.py, no loop principal):

    from metrics.exporter import InferenceMetrics

    metrics = InferenceMetrics(port=9309)

    # a cada predição feita pelo SignPredictor:
    with metrics.time_prediction(kind="static"):
        veredito = predictor.predict_static(pose)
    metrics.record_confidence(kind="static", confidence=veredito["confianca"])

Métricas expostas em /metrics (porta configurável, padrão 9309):
    - libras_inference_confidence{model="static|movement"}       (gauge, última predição)
    - libras_inference_latency_seconds{model="static|movement"}  (histogram)
    - libras_inference_predictions_total{model="static|movement"} (counter)
"""

import time
from contextlib import contextmanager

from prometheus_client import start_http_server, Gauge, Histogram, Counter

METRIC_CONFIDENCE = Gauge(
    "libras_inference_confidence", "Confiança da última predição", ["model"]
)
METRIC_LATENCY = Histogram(
    "libras_inference_latency_seconds", "Tempo de inferência por predição", ["model"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
METRIC_PREDICTIONS_TOTAL = Counter(
    "libras_inference_predictions_total", "Número de predições feitas", ["model"]
)


class InferenceMetrics:
    """Wrapper fino em cima do prometheus_client, só pra não espalhar
    `.labels(model=...)` por todo o translator.py."""

    def __init__(self, port: int = 9309, start_server: bool = True):
        self.port = port
        if start_server:
            start_http_server(port)
            print(f"[metrics] métricas de inferência em http://localhost:{port}/metrics")

    def record_confidence(self, kind: str, confidence: float) -> None:
        METRIC_CONFIDENCE.labels(model=kind).set(confidence)
        METRIC_PREDICTIONS_TOTAL.labels(model=kind).inc()

    @contextmanager
    def time_prediction(self, kind: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            METRIC_LATENCY.labels(model=kind).observe(elapsed)


if __name__ == "__main__":
    # Smoke test: sobe o servidor, registra algumas métricas fake, e
    # deixa rodando alguns segundos pra você conferir em /metrics.
    import random

    metrics = InferenceMetrics(port=9309)
    for _ in range(5):
        kind = random.choice(["static", "movement"])
        with metrics.time_prediction(kind):
            time.sleep(random.uniform(0.01, 0.05))
        metrics.record_confidence(kind, random.uniform(0.3, 0.99))
    print("Métricas de teste registradas. Ctrl+C pra sair.")
    while True:
        time.sleep(1)
