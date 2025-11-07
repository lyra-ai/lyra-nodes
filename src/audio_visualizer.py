from __future__ import annotations

import contextlib
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import colorsys
import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from moviepy.audio.AudioClip import AudioArrayClip  # noqa: F401
except ImportError:
    AudioArrayClip = None  # type: ignore

AudioLike = Union[Dict[str, Any], Tuple[Any, Any], torch.Tensor, list]


def _clamp(v: float, vmin: float = 0.0, vmax: float = 1.0) -> float:
    return max(vmin, min(v, vmax))


class LyraAudioVisualizer:
    CATEGORY = "Lyra/Video"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("video_files",)
    FUNCTION = "render_visualizer"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        return {
            "required": {
                "image": ("IMAGE",),
                "audio_clip": ("AUDIO",),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Output frame rate handed to FFmpeg.",
                }),
                "visualizer_height": ("INT", {
                    "default": 160,
                    "min": 32,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Height in pixels for the waveform strip.",
                }),
                "panel_mode": (["Gradient", "Glass", "Solid", "Disabled"], {
                    "default": "Gradient",
                    "tooltip": "Gradient blends with the image automatically.",
                }),
                "auto_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Derive wave/panel colors from the image palette.",
                }),
                "wave_color": ("STRING", {
                    "default": "#00ffaa",
                    "multiline": False,
                    "tooltip": "Wave color if auto_colors is False.",
                }),
                "panel_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "Panel/tint color (Solid/Gradient when auto_colors is False).",
                }),
                "panel_opacity": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Opacity for Solid/Gradient panel (also tint strength for glass).",
                }),
                "glass_blur_sigma": ("FLOAT", {
                    "default": 18.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "Blur radius when panel_mode=Glass.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "visualizer",
                    "multiline": False,
                    "tooltip": "Base filename for the rendered mp4.",
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True → user/output, False → user/temp.",
                }),
            }
        }

    # ------------------------------------------------------------------------- API

    def render_visualizer(
        self,
        image: torch.Tensor,
        audio_clip: AudioLike,
        fps: int = 30,
        visualizer_height: int = 160,
        panel_mode: str = "Gradient",
        auto_colors: bool = True,
        wave_color: str = "#00ffaa",
        panel_color: str = "#000000",
        panel_opacity: float = 0.35,
        glass_blur_sigma: float = 18.0,
        filename_prefix: str = "visualizer",
        save_output: bool = True,
    ):
        self._verify_ffmpeg()

        frame = self._extract_frame(image)
        width, height = frame.shape[2], frame.shape[1]

        if auto_colors:
            derived_wave, derived_panel = self._derive_palette_colors(frame)
            if derived_wave:
                wave_color = derived_wave
            if derived_panel:
                panel_color = derived_panel

        samples, sample_rate = self._normalize_audio(audio_clip)

        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)

        image_path = self._write_image(frame, temp_dir)
        audio_path = self._write_audio(samples, sample_rate, temp_dir)
        gradient_path: Optional[Path] = None

        output_dir = self._resolve_directory(save_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._compose_filename(output_dir, filename_prefix)

        try:
            if panel_mode.lower() == "gradient":
                gradient_path = self._create_gradient_panel(
                    temp_dir=temp_dir,
                    width=width,
                    bar_height=visualizer_height,
                    panel_color_hex=panel_color,
                    wave_color_hex=wave_color,
                    opacity=panel_opacity,
                )

            self._run_ffmpeg(
                image_path=image_path,
                audio_path=audio_path,
                gradient_path=gradient_path,
                output_path=output_path,
                width=width,
                height=height,
                bar_height=visualizer_height,
                wave_color=wave_color,
                panel_color=panel_color,
                panel_opacity=panel_opacity,
                panel_mode=panel_mode,
                glass_sigma=glass_blur_sigma,
                fps=fps,
            )
        finally:
            image_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            if gradient_path:
                gradient_path.unlink(missing_ok=True)

        return ((bool(save_output), [str(output_path)]),)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _verify_ffmpeg() -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is not available on PATH. Install FFmpeg and ensure it is accessible."
            )

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
                return LyraAudioVisualizer._coerce_samples(samples), int(sample_rate)
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

            return LyraAudioVisualizer._coerce_samples(samples), int(sample_rate)

        if isinstance(candidate, torch.Tensor):
            raise ValueError("Wrap bare audio tensors as (tensor, sample_rate) or dict.")

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
    def _write_image(frame_chw: torch.Tensor, temp_dir: Path) -> Path:
        uuid_str = uuid.uuid4().hex[:8]
        path = temp_dir / f"lyra_vis_image_{uuid_str}.png"
        array = (frame_chw.numpy() * 255.0).round().astype(np.uint8)
        array = np.transpose(array[:3], (1, 2, 0))  # CHW -> HWC
        Image.fromarray(array).save(path)
        return path

    @staticmethod
    def _write_audio(samples: torch.Tensor, sample_rate: int, temp_dir: Path) -> Path:
        import wave

        uuid_str = uuid.uuid4().hex[:8]
        path = temp_dir / f"lyra_vis_audio_{uuid_str}.wav"

        clamped = samples.clamp(-1.0, 1.0).detach().cpu().numpy()
        if clamped.ndim == 1:
            clamped = clamped.reshape(1, -1)
        channels, frames = clamped.shape
        pcm = (clamped * 32767.0).round().astype(np.int16)

        with contextlib.closing(wave.open(path.as_posix(), "wb")) as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.T.tobytes())

        return path

    @staticmethod
    def _resolve_directory(save_output: bool) -> Path:
        base = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        return Path(base)

    @staticmethod
    def _compose_filename(directory: Path, prefix: str) -> Path:
        base = prefix.strip() or "visualizer"
        if base.lower().endswith(".mp4"):
            base = base[:-4] or "visualizer"
        ext = ".mp4"

        candidate = directory / f"{base}{ext}"
        if not candidate.exists():
            return candidate

        index = 1
        while True:
            candidate = directory / f"{base}_{index}{ext}"
            if not candidate.exists():
                return candidate
            index += 1

    def _run_ffmpeg(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        gradient_path: Optional[Path],
        output_path: Path,
        width: int,
        height: int,
        bar_height: int,
        wave_color: str,
        panel_color: str,
        panel_opacity: float,
        panel_mode: str,
        glass_sigma: float,
        fps: int,
    ) -> None:
        import wave

        bar_height = max(32, min(bar_height, height))
        fps = max(10, fps)
        panel_mode = (panel_mode or "Gradient").strip().lower()
        panel_y = height - bar_height
        duration = self._audio_duration(audio_path)

        wave_hex = self._color_to_hex(wave_color)
        panel_hex = self._color_to_hex(panel_color)
        alpha = _clamp(panel_opacity)
        sigma = max(1.0, min(glass_sigma, 100.0))

        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-i", str(audio_path)]

        if panel_mode == "gradient" and gradient_path:
            cmd += ["-loop", "1", "-i", str(gradient_path)]

        if panel_mode == "disabled":
            filter_complex = (
                f"[0:v]scale={width}:{height}[bg];"
                f"[1:a]atrim=duration={duration}"
                f",showwaves=mode=cline:s={width}x{bar_height}:rate={fps}:colors={wave_hex}[vis];"
                f"[bg][vis]overlay=0:{panel_y}"
            )
        elif panel_mode == "solid":
            filter_complex = (
                f"[0:v]scale={width}:{height}[bg];"
                f"color=color={panel_hex}@{alpha:.3f}:size={width}x{bar_height}[panel];"
                f"[1:a]atrim=duration={duration}"
                f",showwaves=mode=cline:s={width}x{bar_height}:rate={fps}:colors={wave_hex}[vis];"
                f"[bg][panel]overlay=0:{panel_y}[bgp];"
                f"[bgp][vis]overlay=0:{panel_y}"
            )
        elif panel_mode == "glass":
            filter_complex = (
                f"[0:v]scale={width}:{height},split[vbase][vglass];"
                f"[vglass]crop=iw:{bar_height}:0:{panel_y},gblur=sigma={sigma},format=rgba,"
                f"colorchannelmixer=aa={alpha:.3f}[glass];"
                f"[vbase][glass]overlay=0:{panel_y}[bgp];"
                f"[1:a]atrim=duration={duration}"
                f",showwaves=mode=cline:s={width}x{bar_height}:rate={fps}:colors={wave_hex}[vis];"
                f"[bgp][vis]overlay=0:{panel_y}"
            )
        else:  # gradient
            filter_complex = (
                f"[0:v]scale={width}:{height}[bg];"
                f"[2:v]scale={width}:{bar_height},format=rgba[grad];"
                f"[bg][grad]overlay=0:{panel_y}[bgp];"
                f"[1:a]atrim=duration={duration}"
                f",showwaves=mode=cline:s={width}x{bar_height}:rate={fps}:colors={wave_hex}[vis];"
                f"[bgp][vis]overlay=0:{panel_y}"
            )

        cmd += [
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    # ----------------------------------------------------------------- palette —

    @staticmethod
    def _color_to_hex(color: Union[str, Tuple[int, int, int]]) -> str:
        if isinstance(color, tuple):
            r, g, b = [max(0, min(int(c), 255)) for c in color]
            return f"{r:02x}{g:02x}{b:02x}"
        color = color.strip().lower()
        if color.startswith("#"):
            color = color[1:]
        color = "".join(ch for ch in color if ch in "0123456789abcdef")
        if len(color) == 3:
            color = "".join(ch * 2 for ch in color)
        if len(color) != 6:
            return "00ffaa"
        return color

    @staticmethod
    def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
        color = color.strip().lower()
        if color.startswith("#"):
            color = color[1:]
        color = "".join(ch for ch in color if ch in "0123456789abcdef")
        if len(color) == 3:
            color = "".join(ch * 2 for ch in color)
        if len(color) != 6:
            return (0, 0, 0)
        return tuple(int(color[i:i + 2], 16) for i in range(0, 6, 2))

    @staticmethod
    def _audio_duration(path: Path) -> float:
        import wave

        with contextlib.closing(wave.open(path.as_posix(), "rb")) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate) if rate > 0 else 0.0

    @staticmethod
    def _derive_palette_colors(frame_chw: torch.Tensor) -> Tuple[Optional[str], Optional[str]]:
        array = (frame_chw.numpy() * 255.0).astype(np.uint8)
        array = np.transpose(array[:3], (1, 2, 0))  # CHW -> HWC

        img = Image.fromarray(array, mode="RGB")
        small = img.resize((128, 128), Image.LANCZOS)

        palette_img = small.convert("P", palette=Image.Palette.ADAPTIVE, colors=6)
        palette = palette_img.getpalette() or []
        counts = palette_img.getcolors() or []

        if not palette or not counts:
            return None, None

        rgb_counts = []
        for count, index in counts:
            base = index * 3
            if base + 2 >= len(palette):
                continue
            r, g, b = palette[base], palette[base + 1], palette[base + 2]
            rgb_counts.append((count, (r, g, b)))

        if not rgb_counts:
            return None, None

        def brightness(rgb_tuple: Tuple[int, int, int]) -> float:
            r, g, b = rgb_tuple
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        sorted_by_count = sorted(rgb_counts, key=lambda x: x[0], reverse=True)

        brightest = max(sorted_by_count, key=lambda item: brightness(item[1]))[1]
        darkest = min(sorted_by_count, key=lambda item: brightness(item[1]))[1]

        wave_rgb = LyraAudioVisualizer._boost_saturation(brightest, factor=1.18)
        panel_rgb = LyraAudioVisualizer._darken_color(darkest, factor=0.58)

        return (
            f"#{wave_rgb[0]:02x}{wave_rgb[1]:02x}{wave_rgb[2]:02x}",
            f"#{panel_rgb[0]:02x}{panel_rgb[1]:02x}{panel_rgb[2]:02x}",
        )

    @staticmethod
    def _boost_saturation(rgb: Tuple[int, int, int], factor: float = 1.2) -> Tuple[int, int, int]:
        r, g, b = [c / 255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = _clamp(s * factor)
        v = _clamp(v * 1.05)
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        return int(r2 * 255), int(g2 * 255), int(b2 * 255)

    @staticmethod
    def _darken_color(rgb: Tuple[int, int, int], factor: float = 0.5) -> Tuple[int, int, int]:
        return tuple(max(0, min(int(c * factor), 255)) for c in rgb)

    # ------------------------------------------------------------- gradient pane —

    def _create_gradient_panel(
        self,
        *,
        temp_dir: Path,
        width: int,
        bar_height: int,
        panel_color_hex: str,
        wave_color_hex: str,
        opacity: float,
    ) -> Path:
        bow = max(16, bar_height)
        panel_rgb = self._hex_to_rgb(panel_color_hex)
        wave_rgb = self._hex_to_rgb(wave_color_hex)

        top_rgb = self._blend_rgb(panel_rgb, wave_rgb, 0.25)
        bottom_rgb = self._blend_rgb(panel_rgb, wave_rgb, 0.05)
        top_alpha = _clamp(opacity * 0.55)
        bottom_alpha = _clamp(opacity)

        gradient = np.zeros((bow, width, 4), dtype=np.uint8)

        for y in range(bow):
            t = y / max(1, bow - 1)
            rgb = self._lerp_tuple(top_rgb, bottom_rgb, t)
            alpha = self._lerp(top_alpha, bottom_alpha, t)
            gradient[y, :, :3] = rgb
            gradient[y, :, 3] = int(alpha * 255)

        gradient_img = Image.fromarray(gradient, mode="RGBA")

        uuid_str = uuid.uuid4().hex[:8]
        path = temp_dir / f"lyra_vis_gradient_{uuid_str}.png"
        gradient_img.save(path)
        return path

    @staticmethod
    def _lerp_tuple(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        return tuple(int((1 - t) * a[i] + t * b[i]) for i in range(3))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return (1 - t) * a + t * b

    @staticmethod
    def _blend_rgb(base: Tuple[int, int, int], accent: Tuple[int, int, int], mix: float) -> Tuple[int, int, int]:
        return tuple(
            int((1 - mix) * base[i] + mix * accent[i])
            for i in range(3)
        )
