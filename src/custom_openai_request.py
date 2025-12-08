"""
Lyra Custom OpenAI Request
==========================

A robust, synchronous OpenAI-compatible API request node.
Designed for maximum compatibility across Python versions and environments.
"""

import json
import ast
import time
import traceback
from typing import Dict, Tuple, Optional, Any, List

try:
    import requests
except ImportError:
    requests = None

class LyraCustomOpenAIRequest:
    CATEGORY = "Lyra/Utility"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("responses_json", "status_code", "success_count")
    FUNCTION = "execute_openai_request"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "model": ("STRING", {"default": "", "multiline": False}),
                "messages": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                }),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 128000}),
                "timeout": ("FLOAT", {"default": 30.0, "min": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "retry_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
            },
            "optional": {
                "system_message": ("STRING", {"default": "EMPTY_INPUT", "multiline": True}),
                "intro_message": ("STRING", {"default": "EMPTY_INPUT", "multiline": True}),
                "prefill_message": ("STRING", {"default": "EMPTY_INPUT", "multiline": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when seed changes."""
        return kwargs.get("seed", 0)

    @staticmethod
    def _safe_str(val: Any) -> str:
        """Safely convert any value to string."""
        if val is None:
            return ""
        try:
            return str(val)
        except Exception:
            return ""

    @staticmethod
    def _sanitize_input(val: Any) -> str:
        """Sanitize input value, handling None and EMPTY_INPUT sentinel."""
        if val is None:
            return ""
        try:
            s = str(val)
        except Exception:
            return ""
        
        stripped = s.strip()
        if stripped == "EMPTY_INPUT" or stripped == "":
            return ""
        return s

    @staticmethod
    def _safe_json_dumps(obj: Any, fallback: str = "[]") -> str:
        """Safely serialize object to JSON string."""
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError):
            try:
                return json.dumps(obj, indent=2, ensure_ascii=True)
            except Exception:
                return fallback

    def _parse_messages(self, raw_msgs: str) -> List[Dict[str, str]]:
        """
        Parse messages string into a list of message dicts.
        Handles various input formats robustly.
        """
        if not raw_msgs or not raw_msgs.strip():
            return []

        raw_msgs = raw_msgs.strip()

        # MyShell Fix: Strip outer braces if wrapping a JSON array
        if raw_msgs.startswith("{") and raw_msgs.endswith("}"):
            inner = raw_msgs[1:-1].strip()
            if inner.startswith("[") and inner.endswith("]"):
                raw_msgs = inner

        # Try JSON parsing first
        try:
            parsed = json.loads(raw_msgs)
            return self._normalize_messages(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try ast.literal_eval as fallback (for Python dict syntax)
        try:
            parsed = ast.literal_eval(raw_msgs)
            return self._normalize_messages(parsed)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            pass

        # Last resort: try to extract JSON-like content
        try:
            # Find first [ and last ]
            start_idx = raw_msgs.find("[")
            end_idx = raw_msgs.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_substr = raw_msgs[start_idx:end_idx + 1]
                parsed = json.loads(json_substr)
                return self._normalize_messages(parsed)
        except Exception:
            pass

        print("[Lyra OpenAI] Warning: Could not parse messages, returning empty list")
        return []

    def _normalize_messages(self, parsed: Any) -> List[Dict[str, str]]:
        """Normalize parsed messages into a proper list of dicts."""
        if parsed is None:
            return []

        # Single dict -> wrap in list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # Must be a list at this point
        if not isinstance(parsed, list):
            return []

        # Filter and validate each message
        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            
            # Ensure role and content exist
            role = item.get("role", "")
            content = item.get("content", "")