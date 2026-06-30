import email
from bs4 import BeautifulSoup
from docx import Document
import fitz


def extract_text_from_pdf(path: str) -> str:
    """
    Extracts text from each page using PyMuPDF.
    Tags each page number for reference.
    """
    text_pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text("text") or ""
        text_pages.append(f"\n\n=== Page {i} ===\n\n{page_text}")
    doc.close()
    return "\n".join(text_pages)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_clean_text_from_eml(file_path: str) -> str:
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f)

    text = ""

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))

        # Ignore attachments
        if "attachment" in content_disposition:
            continue

        # Extract plain text if available
        if content_type == "text/plain":
            charset = part.get_content_charset() or "utf-8"
            try:
                text = part.get_payload(decode=True).decode(charset)
                if text.strip():
                    break  # Prefer plain text if available
            except Exception:
                continue

        # Fallback to extracting from HTML
        elif content_type == "text/html":
            charset = part.get_content_charset() or "utf-8"
            try:
                html = part.get_payload(decode=True).decode(charset)
                soup = BeautifulSoup(html, "html.parser")
                clean_text = soup.get_text(separator="\n")
                if clean_text.strip():
                    text = clean_text
            except Exception:
                continue

    return text.strip()
