# FILE: src/memory/session_manager.py
# PURPOSE: Persist a stable user session identifier across container restarts.

import os
from uuid import uuid4

SESSION_FILE = "/app/session_data/.session_id"


class SessionManager:
    """Persist user_id across Docker restarts."""

    def __init__(self, session_file: str = SESSION_FILE) -> None:
        """Initialize the session manager with a file path."""
        normalized = (session_file or "").strip()
        self.session_file = normalized or SESSION_FILE

    def get_or_create_user_id(self) -> str:
        """Return an existing session id or create and persist a new one."""
        directory = os.path.dirname(self.session_file) or "."
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.session_file):
            with open(self.session_file, "r", encoding="utf-8") as handle:
                user_id = handle.read().strip()
            if user_id:
                print("[Returning session]")
                return user_id
        user_id = str(uuid4())
        with open(self.session_file, "w", encoding="utf-8") as handle:
            handle.write(user_id)
        print("[New session]")
        return user_id
