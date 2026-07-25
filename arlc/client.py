"""
arlc/client.py

Dataset & Evaluation Client for the Legal Hybrid RAG system.
Handles loading benchmark questions (`questions.json`), accessing the local PDF corpus (`docs_corpus`),
and exporting evaluation reports. Can also interact with optional remote evaluation servers if configured.
"""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from arlc.config import EnvConfig, get_config

logger = logging.getLogger(__name__)


class EvaluationClient:
    """
    Client for loading legal RAG benchmark datasets (questions + legal PDFs) and managing evaluation runs.
    Designed primarily as a self-contained local evaluation engine, with optional REST API capabilities.
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config: EnvConfig = config or get_config()
        self.base_url: Optional[str] = self.config.eval_base_url.rstrip("/") if self.config.eval_base_url else None
        self.api_key: Optional[str] = self.config.eval_api_key

        self.headers: Dict[str, str] = {
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

    @classmethod
    def from_env(cls) -> "EvaluationClient":
        """Instantiate a client using environment settings."""
        return cls(get_config())

    def download_questions(self, fallback_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Load evaluation benchmark questions.
        First checks the configured local questions path (or fallback_path).
        If not found and a remote API is configured, fetches from the REST API.
        """
        target_path = Path(fallback_path or self.config.questions_path)

        # 1. Local Dataset Priority (Self-Contained Mode)
        if target_path.exists():
            logger.info("Loading evaluation questions from local dataset: %s", target_path)
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "questions" in data:
                    return data["questions"]
                if isinstance(data, list):
                    return data

        # 2. Optional Remote REST API Fetch
        if self.base_url and self.api_key:
            url = f"{self.base_url}/questions"
            try:
                logger.info("Fetching questions from remote server: %s", url)
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "questions" in data:
                    return data["questions"]
                if isinstance(data, list):
                    return data
            except requests.RequestException as exc:
                logger.warning("Remote API fetch failed: %s", exc)

        raise FileNotFoundError(f"Could not load questions: local file {target_path} does not exist and no remote API available.")

    def download_documents(self, output_dir: Optional[Union[str, Path]] = None, fallback_dir: Union[str, Path] = "public_dataset/docs_corpus") -> Path:
        """
        Locate or unpack the legal PDF corpus directory.
        Checks output_dir and fallback_dir first for existing PDFs before attempting remote download.
        """
        target_dir = Path(output_dir or self.config.docs_dir)

        # 1. Check if local directory exists and has PDFs
        if target_dir.exists() and any(target_dir.glob("*.pdf")):
            logger.info("Using local PDF corpus at: %s", target_dir)
            return target_dir

        fallback_path = Path(fallback_dir)
        if fallback_path.exists() and any(fallback_path.glob("*.pdf")):
            logger.info("Using fallback PDF corpus at: %s", fallback_path)
            return fallback_path

        # 2. Optional Remote ZIP Download
        if self.base_url and self.api_key:
            url = f"{self.base_url}/documents"
            try:
                logger.info("Downloading PDF corpus ZIP from remote server: %s...", url)
                response = requests.get(url, headers=self.headers, stream=True, timeout=120)
                response.raise_for_status()

                target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(target_dir)
                logger.info("Extracted PDF corpus into: %s", target_dir)
                return target_dir
            except requests.RequestException as exc:
                logger.warning("Remote document download failed: %s", exc)

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("No PDF files found in %s or %s. Please place your legal PDFs in %s.", target_dir, fallback_path, target_dir)
        return target_dir

    def submit_submission(self, submission_path: Union[str, Path], code_archive_path: Union[str, Path]) -> Dict[str, Any]:
        """Upload submission results to optional remote evaluation server, or report local completion."""
        sub_file = Path(submission_path)
        code_file = Path(code_archive_path)

        if not sub_file.exists():
            raise FileNotFoundError(f"Submission file not found: {sub_file}")

        # If running purely in self-contained local mode, confirm save and return success summary
        if not self.base_url or not self.api_key:
            logger.info("Running in self-contained mode. Benchmark report saved to %s", sub_file)
            with open(sub_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "status": "LOCAL_COMPLETED",
                "message": f"Successfully evaluated {len(payload.get('answers', []))} questions locally.",
                "file": str(sub_file),
            }

        url = f"{self.base_url}/submissions"
        if not code_file.exists():
            raise FileNotFoundError(f"Missing code archive for remote upload: {code_file}")

        logger.info("Uploading %s and %s to remote server %s...", sub_file, code_file, url)
        upload_headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}

        with open(sub_file, "rb") as sf, open(code_file, "rb") as cf:
            files = {
                "submission_file": (sub_file.name, sf, "application/json"),
                "code_archive": (code_file.name, cf, "application/zip"),
            }
            response = requests.post(url, headers=upload_headers, files=files, timeout=60)
            response.raise_for_status()
            return response.json()
