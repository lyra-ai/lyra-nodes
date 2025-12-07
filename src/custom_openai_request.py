import json
import ast
import time
from typing import Dict, Tuple, Optional

import requests

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
                    "password": True,
                }),
                "model": ("STRING", {"default": "", "multiline": False}),
                "messages": ("STRING", {
                    "default": '[]',
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

    def execute_openai_request(
        self,
        url: str,
        api_key: str,
        model: str,
        messages: str,
        max_tokens: int,
        timeout: float,
        seed: int,
        retry_attempts: int,
        system_message: Optional[str] = "",
        intro_message: Optional[str] = "",
        prefill_message: Optional[str] = "",
    ) -> Tuple[str, str, int]:

        # --- Helper: Sanitize Input ---
        def sanitize(val):
            if val is None:
                return ""
            s = str(val)
            if s.strip() == "EMPTY_INPUT":
                return ""
            return s

        # Apply sanitization
        system_msg = sanitize(system_message)
        intro_msg = sanitize(intro_message)
        prefill_msg = sanitize(prefill_message)
        raw_msgs = sanitize(messages)

        # 1. Parsing Messages
        messages_list = []

        # MyShell Fix: Strip outer braces
        if raw_msgs.startswith("{") and raw_msgs.endswith("}"):
            inner = raw_msgs[1:-1].strip()
            if inner.startswith("[") and inner.endswith("]"):
                raw_msgs = inner

        if raw_msgs:
            try:
                messages_list = json.loads(raw_msgs)
            except json.JSONDecodeError:
                try:
                    messages_list = ast.literal_eval(raw_msgs)
                except (ValueError, SyntaxError):
                    messages_list = []

        if isinstance(messages_list, dict):
            messages_list = [messages_list]
        if not isinstance(messages_list, list):
            messages_list = []

        # 2. Construct Stack
        final_messages = []

        # A. System
        if system_msg.strip():
            final_messages.append({"role": "system", "content": system_msg})

        # B. Intro
        if intro_msg.strip():
            final_messages.append({"role": "user", "content": "."})
            final_messages.append({"role": "assistant", "content": intro_msg})

        # C. History
        final_messages.extend(messages_list)

        # D. Prefill
        if prefill_msg.strip():
            if final_messages:
                last_msg = final_messages[-1]
                last_role = last_msg.get("role", "")
                if last_role != "user":
                    final_messages.append({"role": "user", "content": "."})
            final_messages.append({"role": "assistant", "content": prefill_msg})

        # 3. Payload
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

        # 4. Request (Sync Loop)
        response_text = ""
        status_code = 0

        # Use a session for better connection pooling logic (even in sync mode)
        with requests.Session() as session:
            for attempt in range(retry_attempts + 1):
                try:
                    response = session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    response_text = response.text
                    status_code = response.status_code

                    if status_code == 200 and response_text.strip():
                        break

                except Exception as e:
                    response_text = f"Error: {e}"
                    status_code = 500

                # Retry logic
                if attempt < retry_attempts:
                    time.sleep(1) # Blocking sleep

        responses_json_str = json.dumps([response_text], indent=2)
        status_code_str = str(status_code)
        success_count = 1 if 200 <= status_code < 300 else 0

        return (responses_json_str, status_code_str, success_count)