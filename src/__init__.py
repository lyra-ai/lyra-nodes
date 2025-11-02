from .image_audio_fusion import LyraImageAudioFusion
from .url_audio_loader import LyraLoadAudioFromURL
from .url_image_loader import LyraLoadImageFromURL
from .filename_builder import LyraFilenameBuilder

NODE_CLASS_MAPPINGS = {
    "LyraImageAudioFusion": LyraImageAudioFusion,
    "LyraLoadAudioFromURL": LyraLoadAudioFromURL,
    "LyraLoadImageFromURL": LyraLoadImageFromURL,
    "LyraFilenameBuilder": LyraFilenameBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraImageAudioFusion": "Lyra – 🎞️ Image Audio Fusion",
    "LyraLoadAudioFromURL": "Lyra – 🌐 Load Audio from URL",
    "LyraLoadImageFromURL": "Lyra – 🖼️ Load Image from URL",
    "LyraFilenameBuilder": "Lyra – 🏷️ Build Filename",
}
