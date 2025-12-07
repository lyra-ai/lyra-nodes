"""
Lyra Connectivity Test
======================

A simple, synchronous network test using the 'requests' library.
Useful for debugging if 'httpx' or async loops are crashing your environment.
"""

import requests
from typing import Dict, Tuple

class LyraConnectivityTest:
    CATEGORY = "Lyra/Debug"
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("connected", "status_message")
    FUNCTION = "test_connection"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "url": ("STRING", {
                    "default": "https://www.google.com",
                    "multiline": False,
                    "tooltip": "The URL to ping.",
                }),
                "timeout": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.5,
                }),
            },
        }

    def test_connection(self, url: str, timeout: float) -> Tuple[bool, str]:
        print(f"[Lyra Debug] Testing connection to: {url}")

        try:
            # We use a session to mimic a browser slightly better
            with requests.Session() as s:
                # Basic headers to avoid being blocked immediately by some firewalls
                s.headers.update({
                    "User-Agent": "LyraConnectivityTest/1.0",
                })
                response = s.get(url, timeout=timeout)

                # Check status code
                if 200 <= response.status_code < 300:
                    msg = f"Success: {response.status_code} OK"
                    print(f"[Lyra Debug] {msg}")
                    return (True, msg)
                else:
                    msg = f"Failed: HTTP {response.status_code}"
                    print(f"[Lyra Debug] {msg}")
                    return (False, msg)

        except requests.exceptions.ConnectionError:
            msg = "Error: Connection refused or DNS failure."
            print(f"[Lyra Debug] {msg}")
            return (False, msg)
        except requests.exceptions.Timeout:
            msg = "Error: Request timed out."
            print(f"[Lyra Debug] {msg}")
            return (False, msg)
        except Exception as e:
            msg = f"Error: {str(e)}"
            print(f"[Lyra Debug] {msg}")
            return (False, msg)
