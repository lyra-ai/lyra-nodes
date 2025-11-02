"""
Lyra Load Audio from URL
========================

Download an audio file from an HTTP(S) URL, run sanity checks, and return a
ComfyUI AUDIO payload plus the cached file path.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from pydub import AudioSegment

import folder_paths

_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


class LyraLoadAudioFromURL:
    CATEGORY = "Lyra/Audio"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_clip", "saved_path")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HTTP(S) link to an audio file.",
                }),
            },
            "optional": {
                "max_size_mb": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Reject files larger than this limit.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "url_audio",
                    "multiline": False,
                    "tooltip": "Prefix for the temp file name.",
                }),
            }
        }

    def load(
        self,
        url: str,
        max_size_mb: float = 50.0,
        filename_prefix: str = "url_audio",
    ):
        parsed = self._validate_url(url)
        max_bytes = int(max_size_mb * 1024 * 1024)

        content_type, content_length = self._probe_headers(url)
        extension = self._infer_extension(parsed.path, content_type)

        if content_length and content_length > max_bytes:
            raise ValueError(
                f"Remote file reports size {content_length / (1024 * 1024):.1f} MB "
                f"(limit: {max_size_mb:.1f} MB)."
            )

        temp_path = self._download(url, extension, filename_prefix, max_bytes)
        audio_dict = self._load_audio(temp_path)

        return audio_dict, str(temp_path)

    # --------------------------------------------------------------------- helpers
    @staticmethod
    def _validate_url(url: str):
        if not url:
            raise ValueError("URL cannot be empty.")
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported.")
        if not parsed.netloc:
            raise ValueError("URL is missing a host.")
        return parsed

    @staticmethod
    def _probe_headers(url: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            response = requests.head(url, allow_redirects=True, timeout=15)
            if response.status_code >= 400:
                return None, None
            content_type = response.headers.get("Content-Type")
            length = response.headers.get("Content-Length")
            return content_type, int(length) if length and length.isdigit() else None
        except requests.RequestException:
            return None, None

    @staticmethod
    def _infer_extension(path: str, content_type: Optional[str]) -> str:
        if content_type and content_type.lower().startswith("audio/"):
            subtype = content_type.split("/", 1)[1].split(";")[0].strip()
            if subtype:
                return f".{subtype}"
        for suffix in Path(path).suffixes[::-1]:
            suffix = suffix.lower()
            if suffix in _ALLOWED_EXTENSIONS:
                return suffix
        raise ValueError(
            "Cannot determine file type. Server did not report an audio MIME type "
            "and the URL lacks a known audio extension."
        )

    @staticmethod
    def _download(
        url: str,
        extension: str,
        prefix: str,
        max_bytes: int,
    ) -> Path:
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)

        prefix = prefix.strip() or "url_audio"
        filename = f"{prefix}_{torch.randint(0, 1_000_000, ()).item():06d}{extension}"
        destination = temp_dir / filename

        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1_048_576):  # 1 MB
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            destination.unlink(missing_ok=True)
                            raise ValueError(
                                f"Downloaded data exceeded limit of {max_bytes / (1024 * 1024):.1f} MB."
                            )
                        handle.write(chunk)
        except requests.RequestException as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download audio: {exc}") from exc

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty.")

        return destination

    @staticmethod
    def _load_audio(path: Path) -> Dict:
        if AudioSegment is None:
            raise RuntimeError(
                "pydub is not installed. Run `pip install -r requirements.txt` in comfyui-zluda."
            )

        try:
            segment = AudioSegment.from_file(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to decode audio file: {exc}") from exc

        if segment.frame_rate <= 0:
            raise ValueError("Audio file reports an invalid sample rate.")
        if segment.frame_count() == 0:
            raise ValueError("Audio file contains no samples.")

        samples = np.array(segment.get_array_of_samples())
        channels = segment.channels
        sample_width = segment.sample_width * 8  # bits per sample

        if channels > 1:
            samples = samples.reshape(-1, channels).T
        else:
            samples = samples.reshape(1, -1)

        peak = float(2 ** (sample_width - 1))
        tensor = torch.from_numpy(samples.astype(np.float32) / peak)

        return {
            "samples": tensor,
            "sample_rate": int(segment.frame_rate),
            "path": str(path),
        }
