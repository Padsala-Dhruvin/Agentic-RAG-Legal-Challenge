"""Small file-backed cache for deterministic LLM requests."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class LLMResponseCache:
    """Persist successful LLM responses to avoid duplicate API calls."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._data: Dict[str, str] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {str(k): str(v) for k, v in loaded.items()}
            except (OSError, ValueError):
                self._data = {}

    @staticmethod
    def key(model: str, messages: Any, temperature: float) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
