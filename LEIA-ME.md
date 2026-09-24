# LIBRAS Translator (Android) — como montar o projeto

Não dá pra compilar Kotlin/Gradle no ambiente onde eu gerei esses arquivos
(sem Android SDK), então o caminho mais confiável é: você cria o projeto
vazio pelo próprio Android Studio (ele monta o Gradle wrapper certinho pra
versão do seu SDK) e cola estes arquivos por cima.

## Passo 1 — Criar o projeto
Android Studio → **New Project** → **Empty Views Activity**
- Name: `LibrasTranslator`
- Package name: `com.libras.translator`
- Language: **Kotlin**
- Minimum SDK: **API 26 (Android 8.0)** ou superior

## Passo 2 — Substituir/criar os arquivos
Depois que o projeto abrir, substitua/crie exatamente estes arquivos pelos
que estão nesta pasta:

```
app/build.gradle.kts                                   -> build.gradle.kts
app/src/main/AndroidManifest.xml                        -> AndroidManifest.xml
app/src/main/java/com/libras/translator/MainActivity.kt -> MainActivity.kt
app/src/main/res/layout/activity_main.xml                -> activity_main.xml
app/src/main/res/values/strings.xml                       -> strings.xml (mesclar com o que já existe)
```

## Passo 3 — Sincronizar
Clique em **"Sync Now"** quando o Android Studio avisar que o
`build.gradle.kts` mudou. Ele vai baixar CameraX e OkHttp automaticamente
(precisa de internet na hora do build, mas NÃO em tempo de execução do
app — em uso normal, o app só fala com o seu PC na rede local).

## Passo 4 — Rodar
1. No PC, rode `python stream_server.py` — ele imprime o IP local, tipo:
   `Servidor no ar em ws://192.168.0.42:8765`
2. No celular (mesma rede Wi-Fi do PC), abra o app, digite esse IP e
   porta no campo de conexão, aperte **Conectar**.
3. Aponte a câmera pra mão fazendo o sinal — a letra reconhecida aparece
   grande na tela, e vai se acumulando no histórico embaixo.
4. Os botões BACKSPACE / SPACE / CLEAR são 100% locais (não dependem do
   servidor) — editam o texto direto no celular.

## Por que "cleartext" (http/ws sem criptografia)?
O app conversa com o PC por `ws://` simples (sem TLS), porque é tráfego
dentro da sua própria rede Wi-Fi doméstica — não sai pra internet. Por
padrão o Android bloqueia tráfego não-criptografado; o `AndroidManifest.xml`
já vem com `usesCleartextTraffic="true"` pra permitir isso. Se um dia você
quiser expor isso pela internet (fora da rede local), aí sim vale colocar
TLS de verdade (wss://) na frente, com um proxy tipo Caddy/nginx.

## Limitações desta primeira versão (o que eu NÃO fiz ainda)
- **Sem reconexão automática**: se o Wi-Fi cair, você precisa apertar
  "Conectar" de novo. Dá pra automatizar depois.
- **Sem compensação de rotação da câmera**: se a imagem chegar de lado no
  PC, gire o celular pra paisagem ou eu adiciono correção de rotação no
  próximo ajuste.
- **Qualidade/FPS do JPEG fixos no código** (`JPEG_QUALITY`, taxa de
  envio) — dá pra expor como configuração na tela se você quiser ajustar
  fino depois.
