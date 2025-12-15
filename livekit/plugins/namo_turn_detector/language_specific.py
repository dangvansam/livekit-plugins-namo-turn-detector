import json

import numpy as np
from huggingface_hub import hf_hub_download
from livekit.agents.inference_runner import _InferenceRunner

from .base import NamoModelBase
from .log import logger
from .model import ONNX_FILENAME

def _get_hf_model_repo(language: str) -> str:
    """Get the appropriate Hugging Face model repository for a language."""
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
    return f"videosdk-live/Namo-Turn-Detector-v1-{lang_name}"


SUPPORTED_LANGUAGES = {
    "en",  # English
    "vi",  # Vietnamese
    "zh",  # Chinese
}


class _NamoTurnRunnerLanguageSpecific(_InferenceRunner):
    """Unified inference runner for all language-specific Namo turn detection models."""

    INFERENCE_METHOD = "namo_turn_detection_language_specific"

    def initialize(self) -> None:
        """Initialize and load all supported language models."""
        import onnxruntime as ort
        from transformers import DistilBertTokenizer

        self._models: dict[str, dict] = {}

        logger.info("Loading language-specific turn detection models...")

        # Load all supported language models during initialization
        for language in SUPPORTED_LANGUAGES:
            hf_repo = _get_hf_model_repo(language)

            tokenizer = DistilBertTokenizer.from_pretrained(
                hf_repo, truncation_side="left"
            )
            max_length = 512

            model_path = hf_hub_download(
                repo_id=hf_repo,
                filename=ONNX_FILENAME
            )

            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )

            self._models[language] = {
                "tokenizer": tokenizer,
                "session": session,
                "max_length": max_length,
            }

            logger.info(
                f"{language} model loaded",
                extra={
                    "language": language,
                    "model_repo": hf_repo,
                    "model_filename": ONNX_FILENAME,
                    "max_length": max_length,
                }
            )

        logger.info(f"All {len(self._models)} language-specific models initialized")

    def run(self, data: bytes) -> bytes | None:
        """Execute inference on the input data."""
        try:
            data_json = json.loads(data)
            sentence = data_json["sentence"]
            language = data_json.get("language")

            if not language:
                return json.dumps({"probability": 0.0, "error": "Language not specified"}).encode()

            if language not in SUPPORTED_LANGUAGES:
                return json.dumps({"probability": 0.0, "error": f"Unsupported language: {language}"}).encode()

            if not sentence:
                return json.dumps({"probability": 0.0}).encode()

            # Get pre-loaded model for this language
            if language not in self._models:
                return json.dumps({"probability": 0.0, "error": f"Model not loaded for language: {language}"}).encode()

            model_data = self._models[language]
            tokenizer = model_data["tokenizer"]
            session = model_data["session"]
            max_length = model_data["max_length"]

            # Tokenize input
            inputs = tokenizer(
                sentence.strip(),
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

            input_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            if "token_type_ids" in inputs:
                input_dict["token_type_ids"] = inputs["token_type_ids"]

            # Run inference
            outputs = session.run(None, input_dict)

            logits = outputs[0][0]

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)

            eou_probability = float(probabilities[1])

            logger.debug(
                f"Language-specific [{language}] turn detection inference completed",
                extra={
                    "probability": eou_probability,
                    "language": language,
                    "sentence_length": len(sentence),
                    "sentence": sentence,
                },
            )

            return json.dumps({"probability": eou_probability}).encode()

        except Exception as e:
            logger.error(f"Error during turn detection inference: {e}")
            return json.dumps({"probability": 0.0, "error": str(e)}).encode()


_InferenceRunner.register_runner(_NamoTurnRunnerLanguageSpecific)


def _get_language_runner_method(language: str) -> str:
    """
    Get the inference method for language-specific runner.

    Args:
        language: Language code (e.g., 'en', 'fr', 'es')

    Returns:
        str: The inference method name (same for all languages)
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    return _NamoTurnRunnerLanguageSpecific.INFERENCE_METHOD


class LanguageSpecificModel(NamoModelBase):
    """
    Language-specific Namo Turn Detection model.

    This model is optimized for a specific language and provides better performance
    for that language compared to the multilingual model.

    Supported languages: English (en), Vietnamese (vi), Chinese (zh).
    """

    def __init__(self, *, language: str, threshold: float = 0.7, **kwargs):
        """
        Initialize the language-specific model.

        Args:
            language: Language code (e.g., 'en', 'fr', 'es').
            threshold: The threshold for end-of-utterance detection (0.0-1.0). Defaults to 0.5.
            **kwargs: Additional arguments passed to the base class.
        """
        if language.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )

        super().__init__(threshold=threshold, **kwargs)
        self._language = language.lower()

    @property
    def language(self) -> str:
        """Get the configured language code."""
        return self._language

    @property
    def model(self) -> str:
        return f"namo-{self._language}"

    def _inference_method(self) -> str:
        """Get the inference method for this language."""
        return _get_language_runner_method(self._language)

    async def predict_end_of_turn(
        self,
        chat_ctx,
        *,
        timeout: float | None = 10.0,
    ) -> float:
        """
        Predict the probability of end-of-turn for the given chat context.

        Overrides base method to include language in the inference request.
        """
        import asyncio
        import json
        import time

        start_time = time.time()

        try:
            # Extract the last user message
            sentence = self._get_last_user_message(chat_ctx)

            if not sentence:
                logger.debug("No user message found in chat context")
                return 0.0

            # Prepare data for inference - include language
            json_data = json.dumps({
                "sentence": sentence,
                "language": self._language
            }).encode()

            # Execute inference with timeout
            result = await asyncio.wait_for(
                self._executor.do_inference(self._inference_method(), json_data),
                timeout=timeout,
            )

            # Parse result
            result_json = json.loads(result.decode())
            probability = result_json.get("probability", 0.0)

            duration = time.time() - start_time
            return probability

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Turn detection inference timeout",
                extra={"duration": duration, "timeout": timeout, "language": self._language},
            )
            raise

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                f"Error during turn detection: {e}",
                extra={"duration": duration, "error": str(e), "language": self._language},
            )
            return 0.0

    async def supports_language(self, language: str) -> bool:
        """
        Check if the model supports the given language.

        Args:
            language: Language code to check (e.g., 'en', 'fr', 'es')

        Returns:
            bool: True if this model is configured for the given language
        """
        return language in SUPPORTED_LANGUAGES

    @classmethod
    def supported_languages(cls) -> set[str]:
        """
        Get the set of all supported language codes.

        Returns:
            set[str]: Set of supported language codes
        """
        return SUPPORTED_LANGUAGES
