from typing import Optional
from fastapi import Header, HTTPException

from config import get_settings

settings = get_settings()
EXPECTED_BEARER_TOKEN = settings.bearer_token


async def verify_token(authorization: Optional[str] = Header(None)) -> None:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    if token != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or expired token")
