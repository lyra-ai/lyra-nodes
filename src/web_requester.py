import json
import requests
from typing import Dict, Tuple

class LyraWebRequester:
    CATEGORY = "Lyra/Utility"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("responses_json", "status_code", "success_count")
    FUNCTION = "execute_request"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "url": ("STRING", {
                    "default": "https://api.example.com",
                    "multiline": False,
                }),
                "method": (["POST", "GET", "PUT", "DELETE", "PATCH"],),
                "headers": ("STRING", {
                    "default": '{\n  "Authorization": "Bearer ...",\n  "Content-Type": "application/json"\n}',
                    "multiline": True,
                    "tooltip": "Headers as a valid JSON object.",
                }),
                "json_body": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Request body as a valid JSON object.",
                }),
                "timeout": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 300.0, "step": 1.0,
                    "tooltip": "Request timeout in seconds.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Change this number to force the node to re-run.",
                }),
            },
        }

    def execute_request(
        self,
        url: str,
        method: str,
        headers: str,
        json_body: str,
        timeout: float,
        seed: int,
    ) -> Tuple[str, str, int]:
        # Parse headers
        try:
            headers_dict = json.loads(headers) if headers.strip() else {}
        except json.JSONDecodeError:
            return '["Invalid JSON in headers."]', "400", 0

        # Parse body
        try:
            json_payload = json.loads(json_body) if json_body.strip() else None
        except json.JSONDecodeError:
            return '["Invalid JSON in body."]', "400", 0

        # Execute single request (Blocking)
        try:
            response = requests.request(
                method,
                url,
                headers=headers_dict,
                json=json_payload if method in ["POST", "PUT", "PATCH"] else None,
                timeout=timeout,
            )
            response_text = response.text
            status_code = response.status_code
        except requests.exceptions.RequestException as e:
            response_text = f"Request failed: {type(e).__name__} - {e}"
            status_code = 500
        except Exception as e:
            response_text = f"Unexpected error: {e}"
            status_code = 500

        # Format outputs
        responses_json_str = json.dumps([response_text], indent=2)
        status_code_str = str(status_code)
        success_count = 1 if 200 <= status_code < 300 else 0

        return (responses_json_str, status_code_str, success_count)