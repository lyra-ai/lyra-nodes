"""
Lyra Custom OpenAI Request
==========================

A specialized wrapper around the Web Requester for OpenAI-compatible APIs.
Hardcodes headers and structure, exposing only model, messages, and params.
"""

import json
from typing import Dict, Tuple

import httpx

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
                    "tooltip": "The API endpoint URL.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "tooltip": "",
                }),
                "model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "The model ID string.",
                }),
                "messages": ("STRING", {
                    "default": '[{"role": "user", "content": "Hello!"}]',
                    "multiline": True,
                    "tooltip": "A JSON list of message objects.",
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 128000,
                    "step": 1,
                }),
                "timeout": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 300.0, "step": 1.0,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Force re-run (cache busting).",
                }),
            },
        }

    async def execute_openai_request(
        self,
        url: str,
        api_key: str,
        model: str,
        messages: str,
        max_tokens: int,
        timeout: float,
        seed: int,
    ) -> Tuple[str, str, int]:
        # 1. Validate and Parse Messages
        try:
            messages_list = json.loads(messages)
            if not isinstance(messages_list, list):
                return ('["Error: messages input must be a JSON list of objects."]', "400", 0)
        except json.JSONDecodeError:
            return ('["Error: messages input is not valid JSON."]', "400", 0)

        # 2. Construct Headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 3. Construct Payload
        # We force stream=False because ComfyUI doesn't handle streams well in this context
        payload = {
            "model": model,
            "messages": messages_list,
            "max_tokens": max_tokens,
            "stream": False
        }

        # 4. Execute Request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response_text = response.text
                status_code = response.status_code
            except httpx.RequestError as e:
                response_text = f"Request failed: {type(e).__name__} - {e}"
                status_code = 500
            except Exception as e:
                response_text = f"Unexpected error: {e}"
                status_code = 500

        # 5. Format Output (List of 1 string for compatibility)
        responses_json_str = json.dumps([response_text], indent=2)
        status_code_str = str(status_code)
        success_count = 1 if 200 <= status_code < 300 else 0

        return (responses_json_str, status_code_str, success_count)
