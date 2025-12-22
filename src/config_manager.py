import json
import os
import subprocess
import click
from pathlib import Path

CONFIG_DIR = Path.home() / ".emrage"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_THUNDERBIRD = "ruta a tu carpeta de Thunderbird por defecto"

def get_config():
    if not CONFIG_FILE.exists():
        return None
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def check_dependencies():
    """Comprueba e instala dependencias de sistema y modelos."""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        click.echo("Ollama detectado.")
    except:
        click.echo("Ollama no encontrado. Instalando...")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    try:
        subprocess.run(
                    ["ollama", "show", "llama3.2:1b"], 
                    capture_output=True, 
                    text=True
                )
    except:
        click.echo("Comprobando llama3.2:1b...")
        subprocess.run(["ollama", "pull", "llama3.2:1b"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    models = ["en_core_web_md", "es_core_news_lg", "fr_core_news_sm", 
              "de_core_news_sm", "it_core_news_sm", "pt_core_news_sm", "xx_ent_wiki_sm"]
    
    for model in models:
        click.echo(f"Comprobando {model}...", nl=False)
        
        check_command = [
            "poetry", "run", "python", "-c", 
            f"import spacy; exit(0) if spacy.util.is_package('{model}') else exit(1)"
        ]
        
        result = subprocess.run(check_command, capture_output=True)

        if result.returncode == 0:
            click.secho(" INSTALADO", fg="green")
        else:
            click.secho(" NO ENCONTRADO", fg="yellow")
            click.echo(f"Descargando {model}...")
            
            subprocess.run(
                ["poetry", "run", "python", "-m", "spacy", "download", model],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            click.secho(f"Modelo {model} instalado con éxito.", fg="green")