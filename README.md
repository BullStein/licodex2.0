# LIBRAS - Projeto Unificado

Tudo num lugar só: os scripts originais (webcam + k-NN + Ollama) e a
LIBRAS NN Platform (redes neurais treinadas, na pasta `nn/`).

## Como começar

Duas formas de usar, escolha uma:

**Terminal (menu numerado):**
```
python main.py
```

**Painel web (sem terminal, com status e log ao vivo):**
```
cd panel
pip install flask
python app.py
```
Abre em http://localhost:5000 -- botões pra capturar, treinar, ligar/
desligar o daemon contínuo, e acompanhar acurácia/amostras/log em
tempo real. As janelas de webcam (capture_signatures.py,
photo_capture.py) ainda abrem separadas -- o painel só clica o botão
de "iniciar" por você, não mostra a câmera dentro do navegador.

Ambos conferem sozinhos se cada ação tem o que precisa (arquivos,
checkpoints, dados) antes de tentar rodar.

## Estrutura

```
libras-project/
├── main.py                    <- comece por aqui
│
├── (scripts originais -- webcam, sempre na raiz)
├── capture_signatures.py      captura letras/comandos/movimento
├── photo_capture.py           tira fotos pra imgs/
├── batch_train_from_images.py treina data.json a partir de imgs/ (k-NN)
├── translator.py              tradutor em tempo real (k-NN antigo)
├── auto_validador_ia.py       auto-validador com Ollama (antigo)
├── matching_improved.py       classificador k-NN ponderado
├── dynamic_signatures.py      classificador de movimento (k-NN)
├── ai_judge.py                juiz via Ollama
├── ollama_config.py           config de perfis do Ollama
│
├── (scripts novos -- rede neural, também na raiz)
├── translator_nn.py           tradutor em tempo real (rede neural)
├── auto_validador_nn.py       auto-validador com rede neural
│
├── nn/                        LIBRAS NN Platform (independente)
│   ├── train/                 dataset, modelos, treino, daemon
│   ├── inference/              predictor.py (carrega os checkpoints)
│   ├── metrics/                exportador de métricas Prometheus
│   ├── monitoring/             docker-compose (Grafana + Prometheus)
│   └── checkpoints/             .pt gerados pelo treino (vazio no início)
│
├── panel/                      painel web de controle (Flask)
│   ├── app.py                   servidor -- start/stop de processos, status, log
│   └── templates/index.html      a página em si
│
├── sync_server/                servidor de sincronização (hospede externamente)
│   ├── app.py                    API -- merge de dados, upload de fotos/checkpoints
│   ├── Dockerfile                 pra deploy em Railway/Render/VPS/etc.
│   └── README.md                  instruções de deploy detalhadas
├── sync_client.py              roda em CADA PC -- sincroniza com o sync_server
│
└── imgs/                      (você cria) preset de fotos por letra
```

## Fluxo recomendado

1. **Capturar dados**: opção 1 (webcam, letras/comandos/movimento) e/ou
   opção 2 (fotos pra `imgs/`).
2. **Treinar a rede neural**: opção 6. Gera os checkpoints em
   `nn/checkpoints/`.
3. **Testar em tempo real**: opção 7 (tradutor com a rede) ou opção 8
   (auto-validador, que também alimenta mais dados de treino).
4. **Automatizar o retreino**: opção 9, deixa rodando numa janela
   separada enquanto você continua capturando dados na opção 1 -- ele
   retreina sozinho depois de 30s de silêncio.
5. **Monitorar** (opcional, requer Docker): opção 10.
6. **Sincronizar entre PCs** (pessoal, faculdade, etc.): veja
   `sync_server/README.md` pra hospedar o servidor uma vez, depois
   rode `python sync_client.py` (ou o botão "Sincronizar com o
   servidor" no painel web) em cada PC sempre que for trocar de
   máquina -- envia o que você capturou aqui e traz o que capturou no
   outro PC, sem sobrescrever nada de ninguém.

As opções 3, 4, 5 são as versões *antigas* (k-NN + Ollama), mantidas
pra comparação/backup -- não são necessárias se você for direto pra
rede neural.

## Coisas que ainda dependem de você

- **Letras de movimento (H, J, K, X, Z)**: ainda não foram capturadas.
  Rode a opção 1 e use a captura de movimento antes de treinar a rede
  de movimento.
- **Docker**: só necessário pra opção 10 (Grafana/Prometheus). Tudo o
  resto funciona sem ele.
- **Reconhecimento de movimento em tempo real**: nem o `translator.py`
  original nem o `translator_nn.py` fazem isso ainda -- os dois só
  reconhecem letra estática ao vivo. Letras de movimento hoje só são
  usadas no fluxo de captura/validação, não no tradutor contínuo.
