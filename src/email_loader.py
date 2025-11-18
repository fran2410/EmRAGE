import email
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, getaddresses
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from tqdm import tqdm
import json
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
import pytz


@dataclass
class Email:
    id: str
    message_id: str
    date: Optional[str]
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str] = None
    bcc_addresses: List[str] = None
    subject: str = ""
    body: str = ""
    thread_id: Optional[str] = ""
    in_reply_to: Optional[str] = ""
    references: Optional[str] = ""
    x_filename: Optional[str] = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def get_full_text(self) -> str:
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.from_address:
            parts.append(f"From: {self.from_address}")
        if self.to_addresses:
            parts.append(f"To: {', '.join(self.to_addresses)}")
        if self.body:
            parts.append(f"\nContent:\n{self.body}")
        return "\n".join(parts)

    def get_metadata(self) -> Dict:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "date": self.date,
            "from": self.from_address,
            "to": ", ".join(self.to_addresses) if self.to_addresses else "",
            "subject": self.subject,
            "thread_id": self.thread_id,
        }


class EMLEmailLoader:
    def __init__(self, emails_folder: str):
        self.emails_folder = Path(emails_folder)
        self.emails: List[Email] = []
        self.thread_map: Dict[str, List[str]] = {}

    def load_emails(self) -> List[Email]:

        if not self.emails_folder.exists():
            raise FileNotFoundError(f"La carpeta {self.emails_folder} no existe")

        eml_files = list(self.emails_folder.glob("*.eml"))

        if not eml_files:
            print(f"No se encontraron archivos .eml en {self.emails_folder}")
            return []

        print(f"Encontrados {len(eml_files)} archivos .eml")

        for eml_file in tqdm(eml_files, desc="Procesando emails"):
            try:
                email_obj = self._load_eml_file(eml_file)
                if email_obj and email_obj.body:
                    self.emails.append(email_obj)
                    self._update_thread_map(email_obj)
            except Exception as e:
                print(f"Error procesando {eml_file.name}: {e}")
                continue

        print(f"\n{len(self.emails)} emails procesados correctamente")
        print(f"{len(self.thread_map)} hilos de conversación identificados")

        return self.emails

    def _load_eml_file(self, file_path: Path) -> Optional[Email]:

        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        message_id = msg.get("Message-ID", "")
        if message_id:

            message_id = message_id.strip("<>")

        date_iso = self._process_date(msg.get("Date", ""))

        from_addr = self._extract_from_address(msg)
        to_addrs = self._extract_addresses(msg, "To")
        cc_addrs = self._extract_addresses(msg, "Cc")
        bcc_addrs = self._extract_addresses(msg, "Bcc")

        subject = msg.get("Subject", "")

        in_reply_to = msg.get("In-Reply-To", "")
        if in_reply_to:
            in_reply_to = in_reply_to.strip("<>")

        references = msg.get("References", "")

        thread_id = self._calculate_thread_id(
            message_id, in_reply_to, references, subject
        )

        body = self._extract_body(msg)

        email_id = hashlib.md5(
            (message_id + from_addr + date_iso + file_path.name).encode()
        ).hexdigest()[:12]

        email_obj = Email(
            id=email_id,
            message_id=message_id,
            date=date_iso,
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs if cc_addrs else None,
            bcc_addresses=bcc_addrs if bcc_addrs else None,
            subject=subject,
            body=body,
            thread_id=thread_id,
            in_reply_to=in_reply_to if in_reply_to else None,
            references=references if references else None,
            x_filename=file_path.name,
        )

        return email_obj

    def _extract_from_address(self, msg) -> str:
        from_header = msg.get("From", "")
        if not from_header:
            return ""

        name, addr = parseaddr(from_header)
        return addr if addr else ""

    def _extract_addresses(self, msg, header_name: str) -> List[str]:
        header_value = msg.get(header_name, "")
        if not header_value:
            return []

        addresses = getaddresses([header_value])

        return [addr for name, addr in addresses if addr]

    def _extract_body(self, msg) -> str:
        body = ""

        text_part = msg.get_body(preferencelist=("plain", "html"))

        if text_part:
            content = text_part.get_content()

            if text_part.get_content_type() == "text/html":
                body = self._html_to_text(content)
            else:
                body = content
        else:

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))

                    if "attachment" in content_disposition:
                        continue

                    if content_type == "text/plain":
                        try:
                            body = part.get_content()
                            break
                        except:
                            continue
                    elif content_type == "text/html" and not body:
                        try:
                            html_content = part.get_content()
                            body = self._html_to_text(html_content)
                        except:
                            continue
            else:

                try:
                    body = msg.get_content()
                    if msg.get_content_type() == "text/html":
                        body = self._html_to_text(body)
                except:
                    body = str(msg.get_payload())

        return body.strip() if body else ""

    def _html_to_text(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)
        except:
            return html_content

    def _calculate_thread_id(
        self, message_id: str, in_reply_to: str, references: str, subject: str
    ) -> Optional[str]:

        if in_reply_to:
            return hashlib.md5(in_reply_to.encode()).hexdigest()[:12]

        if references:

            refs = references.strip().split()
            if refs:
                first_ref = refs[0].strip("<>")
                if first_ref:
                    return hashlib.md5(first_ref.encode()).hexdigest()[:12]

        if subject:

            import re

            if re.match(r"^(RE|FW|FWD|Re|Fw|Fwd):\s*", subject, re.IGNORECASE):

                normalized = self._normalize_subject(subject)
                if normalized:
                    return hashlib.md5(normalized.encode()).hexdigest()[:12]

        return None

    def _normalize_subject(self, subject: str) -> str:
        if not subject:
            return ""

        import re

        normalized = subject
        while True:
            new_normalized = re.sub(
                r"^(RE|FW|FWD|Re|Fw|Fwd):\s*", "", normalized, flags=re.IGNORECASE
            )
            if new_normalized == normalized:
                break
            normalized = new_normalized

        return normalized.strip().lower()

    def _update_thread_map(self, email_obj: Email):
        thread_id = email_obj.thread_id

        if thread_id:
            if thread_id not in self.thread_map:
                self.thread_map[thread_id] = []

            self.thread_map[thread_id].append(email_obj.id)

    def _process_date(self, date_str: str) -> str:
        if not date_str:
            return ""

        try:

            parsed_date = parsedate_to_datetime(date_str)

            if parsed_date:

                if parsed_date.tzinfo:
                    parsed_date = parsed_date.astimezone(pytz.UTC)
                else:

                    parsed_date = pytz.UTC.localize(parsed_date)

                return parsed_date.isoformat()
        except Exception as e:
            print(f"No se pudo parsear fecha '{date_str}': {e}")

        return date_str

    def save_processed_emails(self, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        emails_data = [email.to_dict() for email in self.emails]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(emails_data, f, indent=2, ensure_ascii=False)

        print(f"\n Emails guardados en {output_file}")
        print(f" Total emails: {len(self.emails)}")
        print(f" Total threads: {len(self.thread_map)}")

    def get_thread_emails(self, thread_id: str) -> List[Email]:
        if thread_id not in self.thread_map:
            return []

        email_ids = self.thread_map[thread_id]
        return [email for email in self.emails if email.id in email_ids]

    def print_thread_summary(self):
        print("\n=== RESUMEN DE THREADS ===")

        for thread_id, email_ids in sorted(
            self.thread_map.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]:
            thread_emails = self.get_thread_emails(thread_id)
            if thread_emails:
                first_email = thread_emails[0]
                print(f"\nThread {thread_id} ({len(email_ids)} emails)")
                print(f"  Subject: {first_email.subject}")
                print(f"  Participants: {first_email.from_address}")


def quick_test(emails_folder: str, output_json: str = None):

    print("=== PROCESANDO EMAILS .EML ===\n")

    loader = EMLEmailLoader(emails_folder)
    emails = loader.load_emails()

    if emails:
        print(f"\n=== ESTADÍSTICAS ===")
        print(f"  - Total emails: {len(emails)}")
        print(f"  - Emails en threads: {sum(1 for e in emails if e.thread_id)}")
        print(f"  - Emails independientes: {sum(1 for e in emails if not e.thread_id)}")
        print(f"  - Emails con subject: {sum(1 for e in emails if e.subject)}")
        print(f"  - Emails con body: {sum(1 for e in emails if e.body)}")
        print(f"  - Remitentes únicos: {len(set(e.from_address for e in emails))}")

        loader.print_thread_summary()

        if output_json:
            loader.save_processed_emails(output_json)
        else:
            loader.save_processed_emails("../data/processed/emails_processed.json")

    return emails, loader


if __name__ == "__main__":

    emails_folder = "../../Emails"

    emails, loader = quick_test(
        emails_folder, output_json="../data/processed/emails_processed.json"
    )
