import click
import os
from src.config_manager import get_config, save_config, check_dependencies, DEFAULT_THUNDERBIRD
from src.thunderbird_loader import process_inbox_incremental_with_path
from src.embeddings_system import  make_embeddings_and_db_from_emails
from src.rag_engine import start_interactive_session
import json

@click.group()
def main():
    pass

@main.command()
def config():
    click.echo("=== Configuración de EmRAGE ===")
    
    check_dependencies()

    t_path = click.prompt(
        "Ruta a la carpeta de correo (Inbox) de Thunderbird",
        default=DEFAULT_THUNDERBIRD
    )
    
    db_path = click.prompt(
        "¿Dónde quieres guardar las bases de datos?",
        default=str(os.path.join(os.getcwd(), "data/db"))
    )

    new_config = {
        "thunderbird_path": t_path,
        "db_path": db_path,
    }
    
    save_config(new_config)
    click.echo("Configuración guardada en ~/.emrage/config.json")
    click.echo("Buscando nuevos correos en Thunderbird...")
    
    emails = process_inbox_incremental_with_path(t_path, save_to_json=False)
    
    if emails:
        click.echo(f"{len(emails)} correos procesados.")
        
        click.echo("Creando base de datos vectorial...")
        make_embeddings_and_db_from_emails(
            emails=emails,
            emails_db_path=os.path.join(db_path, "emails_vector_db"),
            contacts_db_path=os.path.join(db_path, "contacts_vector_db"),
        )
        click.echo("Base de datos creada exitosamente.")
    else:
        click.echo("No se encontraron correos nuevos para procesar.")

@main.command()
@click.pass_context
def run(ctx):
    cfg = get_config()
    if not cfg:
        click.echo("Error: No hay configuración. Ejecuta 'emrage --config' primero.")
        return

    click.echo("Iniciando motor RAG...")
    start_interactive_session(db_path=os.path.join(cfg['db_path'], "emails_vector_db"),
                              contact_db_path=os.path.join(cfg['db_path'], "contacts_vector_db"))

@main.command()
def update():
    click.echo("Forzando actualización de la base de datos...")
    cfg = get_config()
    if not cfg:
        click.echo("Error: No hay configuración. Ejecuta 'emrage --config' primero.")
        return


    click.echo("Buscando nuevos correos en Thunderbird...")
    emails = process_inbox_incremental_with_path(cfg['thunderbird_path'], save_to_json=False)

    if emails:
        click.echo(f"{len(emails)} correos procesados.")
        
        click.echo("Creando base de datos vectorial...")
        make_embeddings_and_db_from_emails(
            emails=emails,
            emails_db_path=os.path.join(cfg['db_path'], "emails_vector_db"),
            contacts_db_path=os.path.join(cfg['db_path'], "contacts_vector_db"),
        )
        click.echo("Base de datos creada exitosamente.")
    else:
        click.echo("No se encontraron correos nuevos para procesar.")

    