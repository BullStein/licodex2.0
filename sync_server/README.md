# Servidor de Sincronização - LIBRAS

API pra sincronizar dados de captura (JSONs brutos, `imgs/`,
checkpoints) entre vários PCs, sem perder nada de nenhuma máquina --
ver o cabeçalho de `app.py` pra entender a lógica de merge.

## IMPORTANTE: armazenamento persistente

Este servidor guarda tudo em arquivos (`SYNC_STORAGE_DIR`, padrão
`./storage`). **Se você hospedar num serviço com disco EFÊMERO
(reseta a cada redeploy/restart), você perde os dados.**

## Deploy gratuito de verdade: Oracle Cloud "Always Free"

Verificado em setembro de 2026: é a única opção que continua
genuinely grátis para sempre com disco persistente (Railway e Render
deixaram de ter isso -- ver histórico da conversa se precisar dos
detalhes). Duas ressalvas:
- Pede cartão de crédito só pra verificar identidade (retém US$1,
  não cobra nada enquanto você ficar dentro do Always Free).
- A capacidade ARM (Ampere A1) às vezes fica "sem capacidade" na sua
  região no momento de criar a VM -- se acontecer, tente de novo
  mais tarde ou troque de região.

### Passo a passo

1. Crie conta em https://www.oracle.com/cloud/free/ (precisa do cartão
   pra verificação).
2. No console da OCI, crie uma instância **Always Free**:
   - Shape: `VM.Standard.A1.Flex` (ARM, Always Free) ou
     `VM.Standard.E2.1.Micro` (x86, Always Free, mais simples de achar
     capacidade disponível se o ARM estiver esgotado na sua região)
   - Imagem: Ubuntu (a mais recente LTS disponível)
   - Não precisa criar um Block Volume separado -- o boot volume
     Always Free já vem com bastante espaço (200 GB no total,
     compartilhado entre as instâncias Always Free da conta)
3. Na tela de rede, abra a porta 8000 pra entrada (Ingress Rule no
   Security List/Network Security Group associado à VM: origem
   `0.0.0.0/0`, porta TCP `8000`).
4. Conecte via SSH na VM (a OCI te dá o comando pronto, com a chave
   que você baixou na criação) e libere a porta também no firewall do
   próprio Ubuntu (o padrão vem restritivo):
   ```
   sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save   # se disponível; senão, veja doc do Ubuntu da sua versão
   ```
5. Instale o Docker na VM:
   ```
   sudo apt update && sudo apt install -y docker.io
   sudo systemctl enable --now docker
   ```
6. Copie a pasta `sync_server/` pra VM (via `scp` do seu PC, ou `git
   clone` se você versionar o projeto) e rode:
   ```
   cd sync_server
   sudo docker build -t libras-sync-server .
   sudo docker run -d --restart=always -p 8000:8000 \
     -e SYNC_API_TOKEN=escolha-um-token-forte \
     -v /home/ubuntu/libras-sync-data:/data \
     libras-sync-server
   ```
   O `--restart=always` garante que volta sozinho se a VM reiniciar.
   O `-v .../libras-sync-data:/data` é o que torna os dados
   persistentes de verdade (fica gravado no disco da VM, não dentro
   do container).
7. Pegue o IP público da VM (mostrado no console da OCI) -- o
   `server_url` de cada PC vai ser `http://<esse-ip>:8000`.

Depois disso, teste com `curl http://<ip>:8000/health` do seu PC --
se responder `{"ok":true,...}`, está no ar.

## Rodando localmente (teste, sem deploy)

```
cd sync_server
pip install -r requirements.txt
set SYNC_API_TOKEN=segredo123        (Windows cmd)
$env:SYNC_API_TOKEN="segredo123"      (PowerShell)
python app.py
```
Servidor sobe em `http://localhost:8000`. Só serve pra testar na
mesma máquina/rede -- pra usar entre PC pessoal e da faculdade de
verdade, precisa estar hospedado em algum lugar acessível pela
internet (daí o deploy acima).

## Configurando cada PC (cliente)

Na raiz do projeto (`libras-project/`, ao lado de `sync_client.py`),
crie `sync_config.json`:
```json
{
  "server_url": "https://seu-servidor.up.railway.app",
  "api_token": "o-mesmo-token-que-voce-definiu-no-servidor"
}
```

Depois, sempre que quiser sincronizar:
```
python sync_client.py
```
Isso envia o que você capturou localmente (mesclando com o que já
está no servidor, sem apagar nada de outro PC) e baixa o que os
outros PCs já sincronizaram. Rode isso antes de começar a trabalhar
(pra puxar o que o outro PC fez) e depois de capturar coisa nova
(pra subir).

## Endpoints (referência)

| Rota | Método | O que faz |
|---|---|---|
| `/health` | GET | Verifica se o servidor está de pé (sem autenticação) |
| `/merge/<data_raw\|commands_raw\|movement_raw>` | POST | Envia um JSON local, servidor faz a união e devolve o resultado |
| `/download/<nome>` | GET | Baixa o JSON completo do servidor |
| `/imgs/manifest` | GET | Lista `{label: [{filename, sha256}]}` |
| `/imgs/upload` | POST | Form-data: `label`, `filename`, `file` |
| `/imgs/download/<label>/<filename>` | GET | Baixa uma foto |
| `/checkpoints/<prefix>/meta` | GET | Metadados do checkpoint atual no servidor |
| `/checkpoints/<prefix>/upload` | POST | Form-data: `meta` (JSON), `model` (.pt), `labels` (.json) -- só promove se não piorar a acurácia |
| `/checkpoints/<prefix>/download/<model\|labels\|meta>` | GET | Baixa uma parte do checkpoint |

Todas as rotas (exceto `/health`) exigem header
`Authorization: Bearer <SYNC_API_TOKEN>`.
