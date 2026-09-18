"""Domain error - carries an HTTP-ish status so the API layer can map it."""
from __future__ import annotations


class WMSError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status
