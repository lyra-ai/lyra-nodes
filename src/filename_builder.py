"""
Lyra Filename Builder
=====================

Generate a sanitized filename string based on a base name with optional random,
date, and time components. Ideal for nodes that expect a plain string filename.
"""

import re
import secrets
import string
from datetime import datetime
from typing import Dict, List, Tuple


class LyraFilenameBuilder:
    CATEGORY = "Lyra/Utility"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "build"

    _PRESETS: Dict[str, Dict[str, str]] = {
        "Custom": {"base": "", "extension": ""},
        "Video": {"base": "video", "extension": ".mp4"},
        "Audio": {"base": "audio", "extension": ".wav"},
        "Image": {"base": "image", "extension": ".png"},
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "preset": (list(cls._PRESETS.keys()),),
                "base_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Base name when preset is Custom.",
                }),
                "extension": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Include dot (e.g. .mp4). Leave blank for none.",
                }),
            },
            "optional": {
                "add_random": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append an 8-character alphanumeric string.",
                }),
                "add_date": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append YYYYMMDD.",
                }),
                "add_time": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append HHMMSS.",
                }),
                "uppercase": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Uppercase the final filename.",
                }),
            }
        }

    def build(
        self,
        preset: str,
        base_name: str,
        extension: str,
        add_random: bool = True,
        add_date: bool = True,
        add_time: bool = True,
        uppercase: bool = False,
    ) -> Tuple[str]:
        config = self._PRESETS.get(preset, self._PRESETS["Custom"])
        base = config["base"] if preset != "Custom" else base_name
        ext = config["extension"] if preset != "Custom" else extension

        sanitized = self._sanitize(base) or "output"

        parts: List[str] = [sanitized]
        if add_random:
            parts.append(self._random_string(8))
        if add_date or add_time:
            now = datetime.now()
            if add_date:
                parts.append(now.strftime("%Y%m%d"))
            if add_time:
                parts.append(now.strftime("%H%M%S"))

        filename = "-".join(part for part in parts if part)

        ext = self._normalize_extension(ext)
        filename = f"{filename}{ext}"
        if uppercase:
            filename = filename.upper()

        return (filename,)

    @staticmethod
    def _sanitize(text: str) -> str:
        text = text.strip()
        text = text.replace(" ", "_")
        text = re.sub(r"[^0-9A-Za-z._-]", "", text)
        text = re.sub(r"_+", "_", text)
        text = re.sub(r"[._-]+$", "", text)
        return text

    @staticmethod
    def _normalize_extension(ext: str) -> str:
        ext = ext.strip()
        if not ext:
            return ""
        if not ext.startswith("."):
            ext = f".{ext}"
        ext = re.sub(r"[^0-9A-Za-z.]", "", ext)
        return ext.lower()

    @staticmethod
    def _random_string(length: int) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))
