"""
Juiz de IA (Ollama) - LIBRAS
-------------------------------
Usa um modelo local do Ollama pra AUTO-JULGAR o chute feito pelo
classificador geometrico (matching_improved.classify_ranked_weighted),
sem precisar de um humano apertando S/N toda hora.

IMPORTANTE - o que a IA recebe:
    Ela NÃO recebe a imagem nem as coordenadas 3D cruas (LLM de texto
    não é bom em geometria bruta e isso só gastaria tokens à toa).
    Ela recebe o RESULTADO do classificador geométrico -- o ranking de
    candidatos com suas distâncias -- e julga se aquele palpite faz
    sentido (distância baixa + boa margem para o 2º colocado = mais
    confiável). É uma segunda opinião estatística, não uma "visão" do
    gesto.

Troca de modelo (casa <-> notebook): ver ollama_config.py.

Uso básico:
    from ai_judge import judge_prediction
    veredito = judge_prediction(ranking, kind="letra")
    # veredito = {
    #   "veredito": "correto" | "incerto" | "errado" | "indisponivel",
    #   "confianca": 0.0-1.0,
    #   "sugestao": "B" (ou None),
    #   "raciocinio": "..."
    # }
"""

import os
import json
import urllib.request
import urllib.error

from ollama_config import get_active_settings

# Timeout em segundos. Modelos grandes (10GB+) podem demorar bastante na
# primeira chamada depois de um tempo ocioso, porque o Ollama descarrega o
# modelo da memória e precisa recarregar do zero ("cold start"). Ajustável
# sem editar o arquivo via variável de ambiente:
#   set LIBRAS_OLLAMA_TIMEOUT=90        (Windows cmd)
#   $env:LIBRAS_OLLAMA_TIMEOUT="90"      (PowerShell)
try:
    REQUEST_TIMEOUT = float(os.environ.get("LIBRAS_OLLAMA_TIMEOUT", "60"))
except ValueError:
    REQUEST_TIMEOUT = 60


ALVO_DESC = {
    "letra": "letra ESTÁTICA de LIBRAS (uma pose só, sem movimento)",
    "comando": "comando de LIBRAS (BACKSPACE/SPACE/CLEAR)",
    "movimento": "letra de LIBRAS COM MOVIMENTO (gesto dinâmico, ex.: H, J, K, X, Z)",
}


def _build_prompt(ranking, kind, extra_context=""):
    alvo = ALVO_DESC.get(kind, "gesto de LIBRAS")
    linhas_ranking = "\n".join(
        f"{i + 1}. {nome} (distância={dist:.4f})" for i, (nome, dist) in enumerate(ranking[:5])
    )
    top1_nome, top1_dist = ranking[0]
    top2_dist = ranking[1][1] if len(ranking) > 1 else None
    margem_txt = f", margem para o 2º colocado: {(top2_dist - top1_dist):.4f}" if top2_dist is not None else ""

    nota_movimento = (
        "\nComo este é um gesto de MOVIMENTO, a distância mede o quão parecida foi a SEQUÊNCIA de poses ao "
        "longo do tempo (não uma pose única) -- por natureza tende a ter distâncias um pouco maiores que "
        "letras estáticas, então não penalize demais só por isso; olhe principalmente a MARGEM para o 2º colocado."
        if kind == "movimento" else ""
    )

    return f"""Você é um verificador de classificação para um sistema de reconhecimento de {alvo} por landmarks de mão (MediaPipe).

Um classificador geométrico (k-NN por distância euclidiana ponderada por landmark) comparou a pose/sequência atual com a base de dados e retornou este ranking (do mais parecido para o menos parecido; distância MENOR = mais parecido):

{linhas_ranking}

Candidato escolhido pelo classificador: "{top1_nome}" (distância {top1_dist:.4f}{margem_txt}).{nota_movimento}
{extra_context}

Sua tarefa: avaliar SE ESSE PALPITE É CONFIÁVEL, baseado apenas nesses números (você não está vendo a imagem, então não invente detalhes visuais).
Regras de bom senso:
- Distância baixa e margem grande para o 2º colocado -> alta confiança ("correto").
- Distância alta, ou candidatos "empatados" com margem pequena -> baixa confiança ("incerto").
- Se outro candidato do ranking parecer mais plausível que o 1º colocado (raro, mas pode acontecer com margem muito pequena), responda "errado" e sugira esse outro.

Responda APENAS com um JSON válido, sem nenhum texto antes ou depois, exatamente neste formato:
{{"veredito": "correto" | "incerto" | "errado", "confianca": <número de 0.0 a 1.0>, "sugestao": "<nome do candidato mais provável, do ranking acima>", "raciocinio": "<uma frase curta explicando por quê>"}}
"""


def _call_ollama(prompt, model, host):
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        # Desliga o modo "thinking" em modelos que suportam raciocínio
        # estendido (ex.: Qwen3, DeepSeek-R1). Sem isso, esses modelos
        # geram um bloco de reflexão inteiro antes da resposta final,
        # o que é lento à toa pra uma tarefa simples como julgar um
        # ranking de distâncias. Em modelos que não suportam essa opção,
        # o Ollama simplesmente ignora o campo.
        "think": False,
        "options": {"temperature": 0.1},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "")


def judge_prediction(ranking, kind="letra", extra_context="", model=None, profile=None):
    """
    ranking: lista [(nome, distancia), ...] já ordenada (saída de
             matching_improved.classify_ranked_weighted).
    kind: "letra" ou "comando", só pra deixar o prompt mais claro.

    Se o Ollama não responder (offline, modelo não baixado, etc.),
    retorna veredito="indisponivel" -- quem chamou deve tratar isso
    como "cai pro fluxo manual", nunca travar o programa.
    """
    if not ranking:
        return {"veredito": "indisponivel", "confianca": 0.0, "sugestao": None,
                "raciocinio": "Nenhum candidato no ranking (mão não detectada)."}

    cfg_model, host, profile_name = get_active_settings(profile)
    model = model or cfg_model

    prompt = _build_prompt(ranking, kind, extra_context)
    try:
        raw = _call_ollama(prompt, model, host)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"veredito": "indisponivel", "confianca": 0.0, "sugestao": ranking[0][0],
                "raciocinio": f"Ollama não respondeu ({model}@{host}, perfil '{profile_name}'): {e}"}

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fallback: alguns modelos (mesmo com think=False) ainda vazam texto
        # de raciocínio antes/depois do JSON. Tenta extrair só o trecho
        # entre a primeira "{" e a última "}" antes de desistir.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                parsed = None
        else:
            parsed = None

    if parsed is not None:
        parsed.setdefault("sugestao", ranking[0][0])
        parsed.setdefault("confianca", 0.5)
        parsed.setdefault("raciocinio", "")
        if parsed.get("veredito") not in ("correto", "incerto", "errado"):
            parsed["veredito"] = "incerto"
        try:
            parsed["confianca"] = max(0.0, min(1.0, float(parsed["confianca"])))
        except (TypeError, ValueError):
            parsed["confianca"] = 0.5
        return parsed

    return {"veredito": "incerto", "confianca": 0.3, "sugestao": ranking[0][0],
            "raciocinio": f"Resposta do modelo não era JSON válido: {raw[:200]!r}"}


if __name__ == "__main__":
    # Teste rápido/manual: simula um ranking e mostra o julgamento no console.
    exemplo = [("A", 0.0421), ("E", 0.1893), ("S", 0.2510)]
    resultado = judge_prediction(exemplo, kind="letra")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
