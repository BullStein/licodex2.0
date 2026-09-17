"""
Config de modelo Ollama - LIBRAS
---------------------------------
Gerencia qual modelo/host do Ollama usar, com troca automática por
MÁQUINA (detecta o hostname do PC), não por um valor fixo salvo no
arquivo. Isso resolve o problema de usar o mesmo ollama_config.json
em mais de um PC (ex.: via git, pendrive, sync de pasta) e ele
"levar" o perfil errado de uma máquina pra outra: cada PC decide seu
próprio perfil a partir do PRÓPRIO hostname, não do que está escrito
no json.

Perfis são livres -- não tem só "casa"/"notebook" fixos, você
cadastra quantos quiser, com o nome que quiser.

Arquivo gerado automaticamente: ollama_config.json (ao lado deste
arquivo). Formato:

    {
      "fallback_profile": "casa",
      "hostname_map": {
        "DESKTOP-CASA-01": "casa",
        "NOTEBOOK-JOAO":   "notebook"
      },
      "profiles": {
        "casa":     {"model": "qwen3:8b", "host": "http://localhost:11434"},
        "notebook": {"model": "qwen3:4b", "host": "http://localhost:11434"}
      }
    }

ORDEM DE PRIORIDADE pra decidir qual perfil usar (a primeira que
"bater" ganha):
    1. Parâmetro explícito (--profile NOME nos scripts, ou profile=
       passado direto na função em código)
    2. Variável de ambiente LIBRAS_OLLAMA_PROFILE
    3. hostname_map: o hostname DESTA máquina (socket.gethostname())
       está cadastrado? -> usa o perfil mapeado
    4. fallback_profile salvo no json
    5. "casa" (se nem isso existir)

------------------------------------------------------------------
COMANDOS (rode "python ollama_config.py --help" pra ver tudo):

  # 1) Primeira vez em CADA máquina: registra o hostname atual e já
  #    cria/associa um perfil com o modelo que essa máquina deve usar.
  python ollama_config.py --register-here casa --model qwen3:8b
  python ollama_config.py --register-here notebook --model qwen3:4b

  # 2) Ver o que essa máquina está resolvendo agora (útil pra debugar):
  python ollama_config.py --show

  # 3) Gerenciar perfis livremente:
  python ollama_config.py --add-profile trabalho qwen3:1.7b
  python ollama_config.py --add-profile trabalho qwen3:1.7b http://192.168.0.42:11434
  python ollama_config.py --set-model notebook qwen3:1.7b
  python ollama_config.py --remove-profile trabalho

  # 4) Gerenciar o mapa de hostname manualmente (sem trocar de máquina):
  python ollama_config.py --map-hostname NOTEBOOK-JOAO notebook
  python ollama_config.py --unmap-hostname NOTEBOOK-JOAO

  # 5) Override rápido sem editar nada (vale só na sessão do terminal):
  set LIBRAS_OLLAMA_PROFILE=notebook          (Windows cmd)
  $env:LIBRAS_OLLAMA_PROFILE="notebook"        (PowerShell)
  export LIBRAS_OLLAMA_PROFILE=notebook        (Linux/Mac)

  # 6) Ver modelos já baixados no Ollama local:
  python ollama_config.py --list
"""

import os
import json
import socket
import argparse
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "ollama_config.json")
DEFAULT_HOST = "http://localhost:11434"

DEFAULT_CONFIG = {
    "fallback_profile": "casa",
    "hostname_map": {},
    "profiles": {
        "casa": {"model": "qwen3:8b", "host": DEFAULT_HOST},
        # "notebook" fica de exemplo já pronto, mas o hostname_map decide
        # se ele realmente é usado nessa máquina ou não.
        "notebook": {"model": "qwen3:4b", "host": DEFAULT_HOST},
    },
}


# --------------------------------------------------------------------------
# Carregamento / gravação do config
# --------------------------------------------------------------------------

def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Aviso: não foi possível ler {os.path.basename(CONFIG_PATH)}, usando config padrão.")
        return json.loads(json.dumps(DEFAULT_CONFIG))

    cfg.setdefault("fallback_profile", DEFAULT_CONFIG["fallback_profile"])
    cfg.setdefault("hostname_map", {})
    cfg.setdefault("profiles", {})
    # nunca sobrescreve perfis que o usuário já tenha customizado, só garante
    # que existam como referência caso o arquivo tenha sido criado zerado.
    for name, prof in DEFAULT_CONFIG["profiles"].items():
        cfg["profiles"].setdefault(name, prof)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Resolução do perfil ativo
# --------------------------------------------------------------------------

def current_hostname():
    return socket.gethostname()


def resolve_profile(explicit_profile=None, cfg=None):
    """
    Aplica a ordem de prioridade e devolve (nome_do_perfil, origem),
    onde origem é uma string tipo "explícito" / "env" / "hostname" /
    "fallback" -- só pra facilitar debug no --show.
    """
    cfg = cfg or load_config()

    if explicit_profile and explicit_profile in cfg["profiles"]:
        return explicit_profile, "explícito"

    env_profile = os.environ.get("LIBRAS_OLLAMA_PROFILE")
    if env_profile and env_profile in cfg["profiles"]:
        return env_profile, "variável de ambiente LIBRAS_OLLAMA_PROFILE"

    host = current_hostname()
    mapped = cfg["hostname_map"].get(host)
    if mapped and mapped in cfg["profiles"]:
        return mapped, f"hostname_map ('{host}')"

    fallback = cfg.get("fallback_profile")
    if fallback and fallback in cfg["profiles"]:
        return fallback, "fallback_profile"

    if cfg["profiles"]:
        first = next(iter(cfg["profiles"]))
        return first, "primeiro perfil disponível (nenhuma regra bateu)"

    raise RuntimeError("Nenhum perfil cadastrado em ollama_config.json. Rode --add-profile primeiro.")


def get_active_settings(profile=None):
    """Retorna (model, host, profile_name) já resolvidos pra essa máquina."""
    cfg = load_config()
    name, _origin = resolve_profile(profile, cfg)
    prof = cfg["profiles"][name]
    return prof["model"], prof.get("host", DEFAULT_HOST), name


# --------------------------------------------------------------------------
# Edição de perfis / mapa de hostname
# --------------------------------------------------------------------------

def add_or_update_profile(name, model, host=None):
    cfg = load_config()
    cfg["profiles"][name] = {"model": model, "host": host or DEFAULT_HOST}
    save_config(cfg)
    print(f"Perfil '{name}' salvo -> modelo '{model}' @ {host or DEFAULT_HOST}")


def remove_profile(name):
    cfg = load_config()
    if name not in cfg["profiles"]:
        print(f"Perfil '{name}' não existe.")
        return
    del cfg["profiles"][name]
    cfg["hostname_map"] = {h: p for h, p in cfg["hostname_map"].items() if p != name}
    if cfg.get("fallback_profile") == name:
        cfg["fallback_profile"] = next(iter(cfg["profiles"]), None)
    save_config(cfg)
    print(f"Perfil '{name}' removido (e desvinculado de qualquer hostname).")


def set_profile_model(name, model, host=None):
    cfg = load_config()
    if name not in cfg["profiles"]:
        cfg["profiles"][name] = {"host": host or DEFAULT_HOST}
    cfg["profiles"][name]["model"] = model
    if host:
        cfg["profiles"][name]["host"] = host
    save_config(cfg)
    print(f"Perfil '{name}' agora usa o modelo '{model}'.")


def map_hostname(hostname, profile_name):
    cfg = load_config()
    if profile_name not in cfg["profiles"]:
        raise ValueError(f"Perfil '{profile_name}' não existe. Crie com --add-profile primeiro.")
    cfg["hostname_map"][hostname] = profile_name
    save_config(cfg)
    print(f"Hostname '{hostname}' agora usa o perfil '{profile_name}'.")


def unmap_hostname(hostname):
    cfg = load_config()
    if hostname in cfg["hostname_map"]:
        del cfg["hostname_map"][hostname]
        save_config(cfg)
        print(f"Hostname '{hostname}' desvinculado.")
    else:
        print(f"Hostname '{hostname}' não estava mapeado.")


def register_here(profile_name, model=None, host=None):
    """Atalho pra primeira configuração de uma máquina nova: garante que o
    perfil exista (criando/atualizando se --model foi passado) e mapeia o
    hostname ATUAL pra ele."""
    cfg = load_config()
    if model:
        cfg["profiles"][profile_name] = {"model": model, "host": host or DEFAULT_HOST}
    elif profile_name not in cfg["profiles"]:
        raise ValueError(
            f"Perfil '{profile_name}' não existe ainda e nenhum --model foi passado. "
            f"Use: --register-here {profile_name} --model <nome-do-modelo>"
        )
    save_config(cfg)
    host_now = current_hostname()
    map_hostname(host_now, profile_name)
    if not cfg.get("fallback_profile"):
        cfg = load_config()
        cfg["fallback_profile"] = profile_name
        save_config(cfg)
    print(f"Máquina '{host_now}' registrada com o perfil '{profile_name}'.")


def set_fallback(name):
    cfg = load_config()
    if name not in cfg["profiles"]:
        raise ValueError(f"Perfil '{name}' não existe.")
    cfg["fallback_profile"] = name
    save_config(cfg)
    print(f"Perfil de fallback agora é: '{name}'.")


def list_installed_models(host=None):
    """Consulta o Ollama local (GET /api/tags) e lista os modelos já baixados."""
    host = host or get_active_settings()[1]
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Não foi possível consultar {url}: {e}")
        print("Verifique se o Ollama está rodando (comando: 'ollama serve').")
        return []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gerencia perfis/modelos do Ollama por máquina (detecção automática por hostname).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--show", action="store_true", help="Mostra o hostname atual, o perfil resolvido e por quê")
    parser.add_argument("--list", action="store_true", help="Lista modelos instalados no Ollama local")

    parser.add_argument("--register-here", metavar="PERFIL",
                         help="Associa ESTE hostname a um perfil (cria/atualiza o perfil se --model for passado)")
    parser.add_argument("--model", metavar="MODELO", help="Usado junto com --register-here / --add-profile")
    parser.add_argument("--host", metavar="HOST", default=None, help="Host do Ollama (padrão: http://localhost:11434)")

    parser.add_argument("--add-profile", nargs=2, metavar=("NOME", "MODELO"), help="Cria/atualiza um perfil")
    parser.add_argument("--remove-profile", metavar="NOME", help="Remove um perfil")
    parser.add_argument("--set-model", nargs=2, metavar=("PERFIL", "MODELO"), help="Troca o modelo de um perfil existente")
    parser.add_argument("--set-fallback", metavar="NOME", help="Define o perfil usado quando nada mais bate")

    parser.add_argument("--map-hostname", nargs=2, metavar=("HOSTNAME", "PERFIL"), help="Associa um hostname a um perfil manualmente")
    parser.add_argument("--unmap-hostname", metavar="HOSTNAME", help="Remove a associação de um hostname")

    args = parser.parse_args()
    did_something = False

    if args.register_here:
        register_here(args.register_here, model=args.model, host=args.host)
        did_something = True
    if args.add_profile:
        add_or_update_profile(args.add_profile[0], args.add_profile[1], host=args.host)
        did_something = True
    if args.remove_profile:
        remove_profile(args.remove_profile)
        did_something = True
    if args.set_model:
        set_profile_model(args.set_model[0], args.set_model[1], host=args.host)
        did_something = True
    if args.set_fallback:
        set_fallback(args.set_fallback)
        did_something = True
    if args.map_hostname:
        map_hostname(args.map_hostname[0], args.map_hostname[1])
        did_something = True
    if args.unmap_hostname:
        unmap_hostname(args.unmap_hostname)
        did_something = True
    if args.list:
        models = list_installed_models()
        print("Modelos instalados no Ollama:" if models else "Nenhum modelo encontrado.")
        for m in models:
            print(f"  - {m}")
        did_something = True

    if args.show or not did_something:
        cfg = load_config()
        host_now = current_hostname()
        name, origin = resolve_profile(None, cfg)
        prof = cfg["profiles"][name]
        print(f"Hostname desta máquina : {host_now}")
        print(f"Perfil resolvido       : {name}  (motivo: {origin})")
        print(f"Modelo                 : {prof['model']}")
        print(f"Host do Ollama         : {prof.get('host', DEFAULT_HOST)}")
        print(f"\nPerfis cadastrados: {sorted(cfg['profiles'].keys())}")
        print(f"Hostnames mapeados: {cfg['hostname_map'] or '(nenhum ainda)'}")


if __name__ == "__main__":
    main()
