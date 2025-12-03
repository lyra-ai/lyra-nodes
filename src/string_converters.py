"""
Lyra String To Types
====================

A utility node that attempts to cast a string input into Integer, Float,
and List formats. Useful for bridging nodes that are picky about types.
"""

import json
from typing import Dict, Tuple, Any, List

class LyraStringToTypes:
    CATEGORY = "Lyra/Utility"
    RETURN_TYPES = ("INT", "FLOAT", "STRING", "LIST")
    RETURN_NAMES = ("int_value", "float_value", "string_value", "list_value")
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "input_string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "The string to convert. For lists, use JSON format ['a','b'] or comma separation 'a,b'.",
                }),
            },
        }

    def convert(self, input_string: str) -> Tuple[int, float, str, List[Any]]:
        # 1. Handle Integer
        # We try float first to handle cases like "1.0" which int() hates
        try:
            int_val = int(float(input_string))
        except (ValueError, TypeError):
            int_val = 0

        # 2. Handle Float
        try:
            float_val = float(input_string)
        except (ValueError, TypeError):
            float_val = 0.0

        # 3. Handle List (Combo)
        # First try parsing as JSON (e.g., '["a", "b"]')
        # If that fails, assume it's a comma-separated string (e.g., "a, b, c")
        list_val = []
        try:
            parsed = json.loads(input_string)
            if isinstance(parsed, list):
                list_val = parsed
            else:
                list_val = [parsed]
        except (json.JSONDecodeError, TypeError):
            if "," in input_string:
                list_val = [x.strip() for x in input_string.split(",")]
            elif input_string.strip():
                list_val = [input_string.strip()]
            else:
                list_val = []

        return (int_val, float_val, input_string, list_val)
