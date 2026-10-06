"""Pull plain text out of whatever a professor uploads, so it can be embedded."""
import zipfile
from pathlib import Path

from pypdf import PdfReader
import docx


def extract_text(path: str) -> str:
    """Best-effort text extraction. Returns '' rather than raising on bad files —
    a corrupt upload should degrade to 'no text to compare', not crash the request."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            return _extract_pdf(path)
        if ext == ".docx":
            return _extract_docx(path)
        if ext in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return ""


def _extract_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def _extract_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs).strip()


def list_zip_contents(path: str, limit: int = 200) -> list[str]:
    """Just a file listing for display on the project detail page — not used for similarity."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            return names[:limit]
    except Exception:
        return []
