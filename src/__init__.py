from .image_audio_fusion import LyraImageAudioFusion
from .url_audio_loader import LyraLoadAudioFromURL

NODE_CLASS_MAPPINGS = {
    "LyraImageAudioFusion": LyraImageAudioFusion,
    "LyraLoadAudioFromURL": LyraLoadAudioFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraImageAudioFusion": "Lyra – 🎞️ Image Audio Fusion",
    "LyraLoadAudioFromURL": "Lyra – 🌐 Load Audio from URL",
}
