"""
Lyra Load Image from URL
========================

Download an image from an HTTP(S) URL, verify it is safe-ish, cache it in the
temp directory, and return a ComfyUI IMAGE tensor plus the saved file path.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image

import folder_paths

_ALLOWED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


class LyraLoadImageFromURL:
    CATEGORY = "Lyra/Image"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_path")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Must be http(s). Example: https://example.com/image.png",
                }),
            },
            "optional": {
                "max_size_mb": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.5,
                    "max": 200.0,
                    "step": 0.5,
                    "tooltip": "Reject files larger than this limit.",
                }),
                "max_pixels": ("INT", {
                    "default": 16_000_000,
                    "min": 65_536,
                    "max": 268_435_456,
                    "step": 65_536,
                    "tooltip": "Reject images with more total pixels than this.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "url_image",
                    "multiline": False,
                    "tooltip": "Prefix for the cached file in temp.",
                }),
                "convert_to_rgb": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force RGB output (drops alpha).",
                }),
            }
        }

    def load(
        self,
        url: str,
        max_size_mb: float = 20.0,
        max_pixels: int = 16_000_000,
        filename_prefix: str = "url_image",
        convert_to_rgb: bool = True,
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

        downloaded = self._download(url, extension, filename_prefix, max_bytes)
        image_tensor = self._load_image(downloaded, max_pixels, convert_to_rgb)

        return image_tensor, str(downloaded)

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
        if content_type and content_type.lower().startswith("image/"):
            subtype = content_type.split("/", 1)[1].split(";")[0].strip()
            if subtype:
                ext = f".{subtype}"
                if ext.lower() in _ALLOWED_IMAGE_EXTENSIONS:
                    return ext
        for suffix in Path(path).suffixes[::-1]:
            suffix = suffix.lower()
            if suffix in _ALLOWED_IMAGE_EXTENSIONS:
                return suffix
        raise ValueError(
            "Cannot determine image type. Server did not report an image MIME type "
            "and the URL lacks a known image extension."
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

        prefix = prefix.strip() or "url_image"
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
            raise RuntimeError(f"Failed to download image: {exc}") from exc

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty.")

        return destination

    @staticmethod
    def _load_image(path: Path, max_pixels: int, convert_to_rgb: bool) -> torch.Tensor:
        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
                if width * height > max_pixels:
                    raise ValueError(
                        f"Image has {width * height:,} pixels (limit: {max_pixels:,})."
                    )
                if convert_to_rgb:
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
                array = np.asarray(img, dtype=np.float32) / 255.0
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to decode image: {exc}") from exc

        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Decoded image has unsupported shape.")

        tensor = torch.from_numpy(array)
        tensor = tensor.unsqueeze(0)  # -> [1, H, W, C]
        return tensor
