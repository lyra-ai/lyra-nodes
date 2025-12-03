"""
Lyra CDN Uploader
=================

A ComfyUI node that uploads an image to a custom CDN endpoint via a multipart
POST request, handling API key authentication and server responses.
"""

from typing import Dict, Tuple
import io

import numpy as np
import requests
import torch
from PIL import Image

class LyraCdnUploader:
    CATEGORY = "Lyra/IO"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("response_text", "status_code", "filename")
    FUNCTION = "upload_image"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "image": ("IMAGE",),
                "endpoint_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "The full URL for your CDN's upload endpoint.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "tooltip": "",
                }),
            },
        }

    def upload_image(
        self,
        image: torch.Tensor,
        endpoint_url: str,
        api_key: str,
    ) -> Tuple[str, int, str]:
        if not endpoint_url or not api_key:
            return ("Endpoint URL and API Key are required.", 400, "")

        # 1. Convert ComfyUI tensor to PNG bytes in memory
        try:
            pil_image = self._tensor_to_pil(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_bytes = buffer.getvalue()
        except Exception as e:
            return (f"Failed to convert image tensor: {e}", 500, "")

        # 2. Prepare request data
        headers = {
            "master-key": api_key,
        }
        files = {
            "file": ("lyra_upload.png", image_bytes, "image/png"),
        }

        # 3. Send the request
        response_text = ""
        status_code = 0
        filename = ""

        try:
            with requests.post(endpoint_url, headers=headers, files=files, timeout=30) as response:
                status_code = response.status_code
                response_text = response.text

                if response.ok:
                    try:
                        data = response.json()
                        if data.get("success"):
                            filename = data.get("filename", "")
                        else:
                            # Handle cases where success=false or is missing
                            filename = f"Upload failed: {data.get('details', 'Unknown error')}"
                    except requests.exceptions.JSONDecodeError:
                        filename = "Server returned non-JSON success response."
                else:
                    # Non-2xx responses might have error details
                    try:
                        error_data = response.json()
                        details = error_data.get('details', 'No details provided.')
                        filename = f"Error: {details}"
                    except requests.exceptions.JSONDecodeError:
                        filename = "Server returned non-JSON error response."

        except requests.exceptions.RequestException as e:
            response_text = f"Network error: {e}"
            status_code = 503  # Service Unavailable

        return (response_text, status_code, filename)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Expected IMAGE tensor with shape [1, H, W, C].")

        img_array = (tensor[0].numpy() * 255.0).astype(np.uint8)

        if img_array.shape[2] == 4:
            return Image.fromarray(img_array, 'RGBA')
        else:
            return Image.fromarray(img_array, 'RGB')
