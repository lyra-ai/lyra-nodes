from .image_audio_fusion import LyraImageAudioFusion
from .url_audio_loader import LyraLoadAudioFromURL
from .url_image_loader import LyraLoadImageFromURL
from .filename_builder import LyraFilenameBuilder
from .audio_visualizer import LyraAudioVisualizer
from .cdn_uploader import LyraCdnUploader
from .web_requester import LyraWebRequester
from .openai_collector import LyraCollectOpenAIResponse
from .custom_openai_request import LyraCustomOpenAIRequest
from .string_converters import LyraStringToTypes
from .connectivity_test import LyraConnectivityTest
from .character_search import LyraCharacterSearch

NODE_CLASS_MAPPINGS = {
    "LyraImageAudioFusion": LyraImageAudioFusion,
    "LyraLoadAudioFromURL": LyraLoadAudioFromURL,
    "LyraLoadImageFromURL": LyraLoadImageFromURL,
    "LyraFilenameBuilder": LyraFilenameBuilder,
    "LyraAudioVisualizer": LyraAudioVisualizer,
    "LyraCdnUploader": LyraCdnUploader,
    "LyraWebRequester": LyraWebRequester,
    "LyraCollectOpenAIResponse": LyraCollectOpenAIResponse,
    "LyraCustomOpenAIRequest": LyraCustomOpenAIRequest,
    "LyraStringToTypes": LyraStringToTypes,
    "LyraConnectivityTest": LyraConnectivityTest,
    "LyraCharacterSearch": LyraCharacterSearch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraImageAudioFusion": "Lyra – 🎞️ Image Audio Fusion",
    "LyraLoadAudioFromURL": "Lyra – 🌐 Load Audio from URL",
    "LyraLoadImageFromURL": "Lyra – 🖼️ Load Image from URL",
    "LyraFilenameBuilder": "Lyra – 🏷️ Build Filename",
    "LyraAudioVisualizer": "Lyra – 🎚️ Audio Visualizer",
    "LyraCdnUploader": "Lyra – 📡 Upload to CDN",
    "LyraWebRequester": "Lyra – 🔗 Web Requester",
    "LyraCollectOpenAIResponse": "Lyra – 🤖 Collect OpenAI Content",
    "LyraCustomOpenAIRequest": "Lyra – 🧠 Custom OpenAI Request",
    "LyraStringToTypes": "Lyra – 🔄 String to Types",
    "LyraConnectivityTest": "Lyra – 📶 Connectivity Test (Requests)",
    "LyraCharacterSearch": "Lyra – 🔍 Character Search"
}