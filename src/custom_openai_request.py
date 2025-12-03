import json
import asyncio
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
                    "tooltip": "",
                }),
                "messages": ("STRING", {
                    "default": '',
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
                "retry_attempts": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of retries if status != 200 or response is empty.",
                }),
            },
            "optional": {
                "system_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "",
                }),
                "intro_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "",
                }),
                "prefill_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "",
                }),
            }
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
        retry_attempts: int,
        system_message: str = "",
        intro_message: str = "",
        prefill_message: str = "",
    ) -> Tuple[str, str, int]:
        # 1. Validate and Parse Input History
        try:
            input_history = json.loads(messages)
            if not isinstance(input_history, list):
                return ('["Error: messages input must be a JSON list of objects."]', "400", 0)
        except json.JSONDecodeError:
            return ('["Error: messages input is not valid JSON."]', "400", 0)

        # 2. Construct the Final Message Stack
        final_messages = []

        # A. System Message
        if system_message.strip():
            final_messages.append({"role": "system", "content": system_message})

        # B. Intro Message
        # Logic: If intro exists, we must precede it with a User message to maintain flow.
        if intro_message.strip():
            final_messages.append({"role": "user", "content": "."})
            final_messages.append({"role": "assistant", "content": intro_message})

        # C. Main History
        final_messages.extend(input_history)

        # D. Prefill Message
        # Logic: Appended to the end. Must ensure the message before it is NOT an assistant/system.
        if prefill_message.strip():
            # If the stack is empty (weird but possible), just add it.
            # Otherwise, check the last role.
            if final_messages:
                last_msg = final_messages[-1]
                last_role = last_msg.get("role", "")
                if last_role != "user":
                    final_messages.append({"role": "user", "content": "."})

            final_messages.append({"role": "assistant", "content": prefill_message})

        # 3. Construct Headers & Payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": final_messages,
            "max_tokens": max_tokens,
            "stream": False
        }

        # 4. Execute Request with Retries
        response_text = ""
        status_code = 0

        async with httpx.AsyncClient() as client:
            for attempt in range(retry_attempts + 1):
                try:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    response_text = response.text
                    status_code = response.status_code

                    # Check for success conditions: 200 OK AND non-empty content
                    if status_code == 200 and response_text.strip():
                        break

                except httpx.RequestError as e:
                    response_text = f"Request failed: {type(e).__name__} - {e}"
                    status_code = 500
                except Exception as e:
                    response_text = f"Unexpected error: {e}"
                    status_code = 500

                if attempt < retry_attempts:
                    await asyncio.sleep(1)

        # 5. Format Output
        responses_json_str = json.dumps([response_text], indent=2)
        status_code_str = str(status_code)
        success_count = 1 if 200 <= status_code < 300 else 0

        return (responses_json_str, status_code_str, success_count)
