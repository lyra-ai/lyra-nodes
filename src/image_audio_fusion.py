"""
Lyra Image Audio Fusion
=======================

Loop a single image for the length of an audio clip, encode an MP4,
and return a VHS_FILENAMES tuple for automation.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import folder_paths

try:
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.VideoClip import ImageClip
except ImportError:
    AudioArrayClip = None
    ImageClip = None


AudioLike = Union[Dict[str, Any], Tuple[Any, Any], torch.Tensor, list]


class LyraImageAudioFusion:
    CATEGORY = "Lyra/Video"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("video_files",)
    FUNCTION = "render_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio_clip": ("AUDIO",),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Frame rate for the encoded clip."
                }),
                "filename_prefix": ("STRING", {
                    "default": "image_audio_fusion",
                    "multiline": False,
                    "tooltip": "Prefix for the output mp4."
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True → user/output, False → user/temp."
                }),
            }
        }

    def render_video(
        self,
        image: torch.Tensor,
        audio_clip: AudioLike,
        fps: int = 24,
        filename_prefix: str = "image_audio_fusion",
        save_output: bool = True,
    ):
        self._verify_moviepy()

        frame = self._extract_frame(image)
        samples, sample_rate = self._normalize_audio(audio_clip)
        duration = max(samples.shape[-1] / sample_rate, 1.0 / max(1, fps))

        output_dir = self._resolve_directory(save_output)
        output_path = self._compose_filename(output_dir, filename_prefix)

        self._encode_video(frame, fps, duration, samples, sample_rate, output_path)

        return ((bool(save_output), [str(output_path)]),)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_directory(save_output: bool) -> Path:
        base = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        directory = Path(base)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _compose_filename(directory: Path, prefix: str) -> Path:
        sanitized = prefix.strip() or "image_audio_fusion"
        if not sanitized.lower().endswith(".mp4"):
            sanitized += ".mp4"
        return directory / sanitized

    @staticmethod
    def _encode_video(
        frame_chw: torch.Tensor,
        fps: int,
        duration: float,
        samples: torch.Tensor,
        sample_rate: int,
        output_path: Path,
    ) -> None:
        frame_np = (frame_chw.numpy() * 255.0).round().astype(np.uint8)
        frame_np = np.transpose(frame_np[:3], (1, 2, 0))
        if frame_np.shape[-1] == 1:
            frame_np = np.repeat(frame_np, 3, axis=-1)

        clip = ImageClip(frame_np).set_duration(duration).set_fps(max(1, fps))

        audio_clip = None
        if samples is not None and sample_rate:
            audio_np = LyraImageAudioFusion._prepare_audio_array(samples.numpy())
            audio_clip = AudioArrayClip(audio_np, fps=sample_rate)
            clip = clip.set_audio(audio_clip)

        clip.write_videofile(
            str(output_path),
            fps=max(1, fps),
            codec="libx264",
            audio_codec="aac" if audio_clip is not None else None,
            audio_bitrate="192k" if audio_clip is not None else None,
            verbose=False,
            logger=None,
        )

        clip.close()
        if audio_clip is not None:
            audio_clip.close()

    @staticmethod
    def _prepare_audio_array(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)

        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        else:
            sample_axis = int(np.argmax(array.shape))
            if sample_axis != array.ndim - 1:
                array = np.moveaxis(array, sample_axis, -1)
            if array.ndim > 2:
                array = array.reshape(-1, array.shape[-1])
            if array.shape[1] > array.shape[0]:
                array = array.T

        array = np.clip(array, -1.0, 1.0)

        if array.ndim == 2 and array.shape[1] > array.shape[0]:
            array = array.T

        return array.astype(np.float32)

    @staticmethod
    def _extract_frame(image: torch.Tensor) -> torch.Tensor:
        tensor = image[0] if isinstance(image, list) else image
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("Expected IMAGE tensor shaped [H, W, C].")
        return tensor.permute(2, 0, 1).contiguous().float().clamp(0.0, 1.0)

    @staticmethod
    def _normalize_audio(audio_clip: AudioLike) -> Tuple[torch.Tensor, int]:
        candidate = audio_clip

        if isinstance(candidate, (list, tuple)):
            if not candidate:
                raise ValueError("AUDIO input is empty.")
            if isinstance(candidate[0], dict):
                candidate = candidate[0]
            elif len(candidate) >= 2:
                samples, sample_rate = candidate[0], candidate[1]
                return LyraImageAudioFusion._coerce_samples(samples), int(sample_rate)
            else:
                candidate = candidate[0]

        if isinstance(candidate, dict):
            sample_keys = ("samples", "audio", "waveform", "data")
            rate_keys = ("sample_rate", "sr", "rate")

            samples = next((candidate.get(k) for k in sample_keys if k in candidate), None)
            sample_rate = next((candidate.get(k) for k in rate_keys if k in candidate), None)

            if samples is None or sample_rate is None:
                raise ValueError(
                    f"AUDIO payload missing samples/rate. Keys: {list(candidate.keys())}"
                )

            return LyraImageAudioFusion._coerce_samples(samples), int(sample_rate)

        if isinstance(candidate, torch.Tensor):
            raise ValueError("Wrap bare tensors as (tensor, sample_rate) or dict.")

        raise ValueError(f"Unsupported AUDIO payload type: {type(candidate)}")

    @staticmethod
    def _coerce_samples(samples: Any) -> torch.Tensor:
        tensor = torch.as_tensor(samples, dtype=torch.float32)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            sample_axis = int(torch.argmax(torch.tensor(tensor.shape)))
            if sample_axis != tensor.ndim - 1:
                tensor = tensor.movedim(sample_axis, -1)

            if tensor.ndim > 2:
                tensor = tensor.reshape(-1, tensor.shape[-1])

            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)

            if tensor.ndim == 2 and tensor.shape[0] > tensor.shape[1]:
                tensor = tensor.transpose(0, 1)

        return tensor.contiguous().float().clamp(-1.0, 1.0)

    @staticmethod
    def _verify_moviepy() -> None:
        if AudioArrayClip is None or ImageClip is None:
            raise RuntimeError(
                "moviepy is missing. Run `pip install -r requirements.txt` inside comfyui-zluda."
            )
