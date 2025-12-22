import mailbox
import json
from pathlib import Path
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
from typing import List, Optional
from src.email_loader import EMLEmailLoader, Email

INBOX_PATH = Path(
    "/home/fran/snap/thunderbird/common/.thunderbird/"
    "5nxd89yk.default-release/Mail/correo.alumnos.upm.es/Inbox"
)
CONFIG_DIR = Path.home() / ".emrage"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATE_FILE = CONFIG_DIR / "processed_message_ids.json"


def load_processed_ids() -> set:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_ids(ids: set):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2)


def mbox_msg_to_email_message(mbox_msg) -> EmailMessage:
    raw_bytes = mbox_msg.as_bytes()
    return BytesParser(policy=policy.default).parsebytes(raw_bytes)


def process_inbox_incremental(save_to_json: bool = False) -> List[Email]:
    processed_ids = load_processed_ids()
    new_processed_ids = set(processed_ids)
    new_emails = []

    loader = EMLEmailLoader(emails_folder="DUMMY")

    mbox = mailbox.mbox(INBOX_PATH)

    for mbox_msg in mbox:
        msg = mbox_msg_to_email_message(mbox_msg)

        message_id = msg.get("Message-ID", "")
        if not message_id:
            continue

        message_id = message_id.strip("<>")

        if message_id in processed_ids:
            continue

        try:
            email_obj: Email = loader._load_eml_file_from_message(
                msg, source_filename="Inbox"
            )
        except Exception as e:
            print(f"Error procesando mensaje {message_id}: {e}")
            continue

        if not email_obj or not email_obj.body:
            continue

        new_emails.append(email_obj)
        new_processed_ids.add(message_id)

    save_processed_ids(new_processed_ids)

    if save_to_json and new_emails:
        output_file = Path("data/processed/new_emails.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([email.to_dict() for email in new_emails], f, indent=2, ensure_ascii=False)
        print(f"Emails guardados en: {output_file}")

    print(f"Nuevos emails procesados: {len(new_emails)}")
    print(f"Total Message-ID almacenados: {len(new_processed_ids)}")

    return new_emails


def process_inbox_incremental_with_path(inbox_path: Path, save_to_json: bool = False) -> List[Email]:
    global INBOX_PATH
    INBOX_PATH = inbox_path
    return process_inbox_incremental(save_to_json)

if __name__ == "__main__":
    process_inbox_incremental(True)
