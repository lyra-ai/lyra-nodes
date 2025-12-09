from .image_audio_fusion import LyraImageAudioFusion
from .url_audio_loader import LyraLoadAudioFromURL
from .url_image_loader import LyraLoadImageFromURL
from .filename_builder import LyraFilenameBuilder
from .audio_visualizer import LyraAudioVisualizer
from .cdn_uploader import LyraCdnUploader
from .string_converters import LyraStringToTypes
from .character_search import LyraCharacterSearch

NODE_CLASS_MAPPINGS = {
    "LyraImageAudioFusion": LyraImageAudioFusion,
    "LyraLoadAudioFromURL": LyraLoadAudioFromURL,
    "LyraLoadImageFromURL": LyraLoadImageFromURL,
    "LyraFilenameBuilder": LyraFilenameBuilder,
    "LyraAudioVisualizer": LyraAudioVisualizer,
    "LyraCdnUploader": LyraCdnUploader,
    "LyraStringToTypes": LyraStringToTypes,
    "LyraCharacterSearch": LyraCharacterSearch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraImageAudioFusion": "Lyra – 🎞️ Image Audio Fusion",
    "LyraLoadAudioFromURL": "Lyra – 🌐 Load Audio from URL",
    "LyraLoadImageFromURL": "Lyra – 🖼️ Load Image from URL",
    "LyraFilenameBuilder": "Lyra – 🏷️ Build Filename",
    "LyraAudioVisualizer": "Lyra – 🎚️ Audio Visualizer",
    "LyraStringToTypes": "Lyra – 🔄 String to Types",
    "LyraCharacterSearch": "Lyra – 🔍 Character Search"
}