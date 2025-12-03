"""
Lyra OpenAI Response Collector
==============================

Parses a list of JSON responses, extracts the assistant's message, and
splits it into 'Clean Content' and 'Chain of Thought' using specific regex patterns.
"""

import json
import re
from typing import Dict, Tuple

class LyraCollectOpenAIResponse:
    CATEGORY = "Lyra/Utility"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "chain_of_thought")
    FUNCTION = "collect"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "responses_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "The output from Lyra Web Requester.",
                }),
            },
        }

    def collect(self, responses_json: str) -> Tuple[str, str]:
        if not responses_json:
            return ("", "")

        try:
            # 1. Parse the outer list
            responses_list = json.loads(responses_json)
            if not isinstance(responses_list, list) or not responses_list:
                return ("Error: Input is not a valid list.", "")

            # 2. Get the last response
            last_response_str = responses_list[-1]
            data = json.loads(last_response_str)

            # 3. Extract raw content
            raw_content = ""
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                raw_content = message.get("content", "")
            elif "error" in data:
                return (f"API Error: {data['error'].get('message', 'Unknown')}", "")
            else:
                return ("Error: Missing 'choices' in response.", "")

            if not raw_content:
                return ("", "")

            # 4. Regex Magic
            # Pattern 1: Standard pair tags <think>...</think> (and variations)
            # Matches horizontal whitespace around content inside tags
            cot_pattern = r"<[|]?.hink[|]?>[^\S\r\n]*([\s\S]*?)[^\S\r\n]*<[|]?\/.hink[|]?>"

            # Pattern 2: Missing start tag (Fallout fallback)
            # Matches from start of string (or near it) until closing tag
            missing_start_pattern = r"^[^\S\r\n]*([\s\S]*?)[^\S\r\n]*<[|]?\/.hink[|]?>"

            chain_of_thought = ""
            cleaned_content = raw_content

            # Attempt to find standard CoT
            match = re.search(cot_pattern, raw_content, re.IGNORECASE)
            if match:
                chain_of_thought = match.group(1)
                # Remove the entire block
                cleaned_content = re.sub(cot_pattern, "", raw_content, flags=re.IGNORECASE)
            else:
                # Attempt fallback for missing start tag
                # We only check this if the standard one didn't fire to avoid false positives
                match_partial = re.search(missing_start_pattern, raw_content, re.IGNORECASE)
                if match_partial:
                    chain_of_thought = match_partial.group(1)
                    cleaned_content = re.sub(missing_start_pattern, "", raw_content, flags=re.IGNORECASE)

            # 5. Final Cleanup (TypeScript reference parity)
            # Remove any stray orphan tags just in case
            cleaned_content = re.sub(r"<[|]?.hink[|]?>", "", cleaned_content, flags=re.IGNORECASE)
            cleaned_content = re.sub(r"<[|]?\/.hink[|]?>", "", cleaned_content, flags=re.IGNORECASE)

            # Trim leading newlines/whitespace so the text starts nice and flush
            cleaned_content = cleaned_content.lstrip()

            return (cleaned_content, chain_of_thought)

        except json.JSONDecodeError:
            return ("Error: Failed to decode JSON.", "")
        except Exception as e:
            return (f"Error parsing: {str(e)}", "")
