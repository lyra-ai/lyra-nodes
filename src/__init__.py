from .image_audio_fusion import LyraImageAudioFusion
from .url_audio_loader import LyraLoadAudioFromURL
from .url_image_loader import LyraLoadImageFromURL

NODE_CLASS_MAPPINGS = {
    "LyraImageAudioFusion": LyraImageAudioFusion,
    "LyraLoadAudioFromURL": LyraLoadAudioFromURL,
    "LyraLoadImageFromURL": LyraLoadImageFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraImageAudioFusion": "Lyra – 🎞️ Image Audio Fusion",
    "LyraLoadAudioFromURL": "Lyra – 🌐 Load Audio from URL",
    "LyraLoadImageFromURL": "Lyra – 🖼️ Load Image from URL",
}
