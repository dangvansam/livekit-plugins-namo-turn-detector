from typing import Optional

from huggingface_hub import hf_hub_download
from .log import logger
from .model import ONNX_FILENAME


def download_model(overwrite_existing: bool = False, language: Optional[str] = None):
    """
    Pre-download the Namo Turn Detection v1 model and tokenizer.

    Args:
        overwrite_existing: If True, re-download even if files exist
        language: Optional language code (e.g., 'en', 'fr'). If None, downloads multilingual model.
    """
    from transformers import AutoTokenizer, DistilBertTokenizer

    if language is None:
        hf_repo = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
        logger.info(f"Downloading multilingual model from {hf_repo}")
        AutoTokenizer.from_pretrained(hf_repo)
        hf_hub_download(repo_id=hf_repo, filename=ONNX_FILENAME)
        logger.info("[✓] Download multilingual model successfully")
    else:
        language_map = {
            "ar": "Arabic",
            "bn": "Bengali",
            "zh": "Chinese",
            "da": "Danish",
            "nl": "Dutch",
            "de": "German",
            "en": "English",
            "fi": "Finnish",
            "fr": "French",
            "hi": "Hindi",
            "id": "Indonesian",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "mr": "Marathi",
            "no": "Norwegian",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian",
            "es": "Spanish",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "vi": "Vietnamese",
        }
        lang_name = language_map.get(language.lower(), language.capitalize())
        hf_repo = f"videosdk-live/Namo-Turn-Detector-v1-{lang_name}"
        logger.info(f"Downloading {lang_name} model from {hf_repo}")
        DistilBertTokenizer.from_pretrained(hf_repo)
        hf_hub_download(repo_id=hf_repo, filename=ONNX_FILENAME)
        logger.info(f"[✓] Download {lang_name} model successfully")
