import os
import tempfile
import uuid
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
import requests
import structlog

logger = structlog.get_logger(__name__)


def download_file(url: str, doc_id: str = "unknown") -> Tuple[str, str]:
    """Download a document URL to a unique temporary file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    parsed = urlparse(url)
    file_ext = os.path.splitext(parsed.path)[1] or ".bin"
    local_path = Path(tempfile.gettempdir()) / f"rag_{doc_id}_{uuid.uuid4().hex}{file_ext}"

    with local_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("file_downloaded", path=str(local_path), extension=file_ext)
    return str(local_path), file_ext.lstrip(".").lower()
