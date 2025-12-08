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
    def INPUT_TYPES(cls):
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

    def _safe_str(self, val):
        """Safely convert any value to string."""
        if val is None:
            return ""
        try:
            return str(val)
        except Exception:
            return ""

    def _sanitize_input(self, val):
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

    def _safe_json_dumps(self, obj, fallback="[]"):
        """Safely serialize object to JSON string."""
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError):
            try:
                return json.dumps(obj, indent=2, ensure_ascii=True)
            except Exception:
                return fallback

    def _parse_messages(self, raw_msgs):
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

    def _normalize_messages(self, parsed):
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

            if not role or not isinstance(role, str):
                continue

            # Normalize content
            if content is None:
                content = ""
            elif not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    content = ""

            result.append({"role": role, "content": content})

        return result

    def _build_message_stack(self, system_msg, intro_msg, messages_list, prefill_msg):
        """Build the final message stack with proper ordering."""
        final_messages = []

        # A. System message
        if system_msg and system_msg.strip():
            final_messages.append({"role": "system", "content": system_msg})

        # B. Intro message (with placeholder user message)
        if intro_msg and intro_msg.strip():
            final_messages.append({"role": "user", "content": "."})
            final_messages.append({"role": "assistant", "content": intro_msg})

        # C. History messages
        if messages_list:
            final_messages.extend(messages_list)

        # D. Prefill message
        if prefill_msg and prefill_msg.strip():
            if final_messages:
                last_msg = final_messages[-1]
                last_role = last_msg.get("role", "") if isinstance(last_msg, dict) else ""
                if last_role != "user":
                    final_messages.append({"role": "user", "content": "."})
            else:
                final_messages.append({"role": "user", "content": "."})

            final_messages.append({"role": "assistant", "content": prefill_msg})

        return final_messages

    def _make_request(self, url, headers, payload, timeout, retry_attempts):
        """
        Make HTTP request with retry logic.
        Returns (response_text, status_code).
        """
        if requests is None:
            return ('{"error": "requests library not installed"}', 500)

        response_text = ""
        status_code = 0
        last_error = ""

        for attempt in range(retry_attempts + 1):
            session = None
            try:
                print("[Lyra OpenAI] Attempt {}/{} to {}".format(
                    attempt + 1, retry_attempts + 1, url
                ))

                session = requests.Session()
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=float(timeout),
                )
                response_text = response.text
                status_code = int(response.status_code)

                print("[Lyra OpenAI] Response status: {}".format(status_code))

                if 200 <= status_code < 300:
                    if response_text and response_text.strip():
                        return (response_text, status_code)
                    else:
                        last_error = "Empty response body"
                else:
                    last_error = "HTTP {}: {}".format(status_code, response_text[:200])

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                status_code = 408
                print("[Lyra OpenAI] Timeout on attempt {}".format(attempt + 1))

            except requests.exceptions.ConnectionError as e:
                last_error = "Connection error: {}".format(self._safe_str(e))
                status_code = 503
                print("[Lyra OpenAI] Connection error on attempt {}: {}".format(attempt + 1, e))

            except requests.exceptions.RequestException as e:
                last_error = "Request error: {}".format(self._safe_str(e))
                status_code = 500
                print("[Lyra OpenAI] Request error on attempt {}: {}".format(attempt + 1, e))

            except Exception as e:
                last_error = "Unexpected error: {}".format(self._safe_str(e))
                status_code = 500
                print("[Lyra OpenAI] Unexpected error on attempt {}: {}".format(attempt + 1, e))
                traceback.print_exc()

            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass

            # Retry delay
            if attempt < retry_attempts:
                sleep_time = min(2 ** attempt, 10)
                print("[Lyra OpenAI] Retrying in {}s...".format(sleep_time))
                time.sleep(sleep_time)

        if not response_text:
            response_text = self._safe_json_dumps({"error": last_error})

        if status_code == 0:
            status_code = 500

        return (response_text, status_code)

    def execute_openai_request(
        self,
        url,
        api_key,
        model,
        messages,
        max_tokens,
        timeout,
        seed,
        retry_attempts,
        system_message=None,
        intro_message=None,
        prefill_message=None,
    ):
        """
        Execute OpenAI-compatible API request.
        Returns (responses_json, status_code, success_count).
        """
        print("[Lyra OpenAI] Starting request (seed: {})".format(seed))

        # Validate required inputs
        url = self._safe_str(url).strip()
        api_key = self._safe_str(api_key).strip()
        model = self._safe_str(model).strip()

        if not url:
            error_response = self._safe_json_dumps([{"error": "URL is required"}])
            return (error_response, "400", 0)

        if not api_key:
            error_response = self._safe_json_dumps([{"error": "API key is required"}])
            return (error_response, "400", 0)

        if not model:
            error_response = self._safe_json_dumps([{"error": "Model is required"}])
            return (error_response, "400", 0)

        # Sanitize optional inputs
        system_msg = self._sanitize_input(system_message)
        intro_msg = self._sanitize_input(intro_message)
        prefill_msg = self._sanitize_input(prefill_message)
        raw_msgs = self._sanitize_input(messages)

        # Parse messages
        messages_list = self._parse_messages(raw_msgs)

        # Build message stack
        final_messages = self._build_message_stack(
            system_msg, intro_msg, messages_list, prefill_msg
        )

        print("[Lyra OpenAI] Final message count: {}".format(len(final_messages)))

        if not final_messages:
            error_response = self._safe_json_dumps([{"error": "No messages to send"}])
            return (error_response, "400", 0)

        # Build request
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }

        payload = {
            "model": model,
            "messages": final_messages,
            "max_tokens": int(max_tokens),
            "stream": False
        }

        # Ensure numeric types
        try:
            timeout = float(timeout)
        except (ValueError, TypeError):
            timeout = 30.0

        try:
            retry_attempts = int(retry_attempts)
        except (ValueError, TypeError):
            retry_attempts = 2

        # Make request
        response_text, status_code = self._make_request(
            url, headers, payload, timeout, retry_attempts
        )

        # Format output
        responses_json_str = self._safe_json_dumps([response_text])
        status_code_str = self._safe_str(status_code)
        success_count = 1 if 200 <= status_code < 300 else 0

        print("[Lyra OpenAI] Complete. Status: {}, Success: {}".format(
            status_code, success_count
        ))

        return (responses_json_str, status_code_str, success_count)