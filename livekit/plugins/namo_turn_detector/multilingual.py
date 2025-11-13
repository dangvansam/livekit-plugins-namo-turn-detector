import json

import numpy as np
from huggingface_hub import hf_hub_download
from livekit.agents.inference_runner import _InferenceRunner

from .base import NamoModelBase
from .log import logger
from .model import ONNX_FILENAME


SUPPORTED_LANGUAGES = {
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


class _NamoTurnRunnerMultilingual(_InferenceRunner):
    """Inference runner for multilingual Namo turn detection model."""

    INFERENCE_METHOD = "namo_turn_detection_multilingual"

    def initialize(self) -> None:
        """Initialize the ONNX model and tokenizer (lazy loading)."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        hf_repo = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"

        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_repo, truncation_side="left"
        )
        self._max_length = 8192

        model_path = hf_hub_download(repo_id=hf_repo, filename=ONNX_FILENAME)

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        logger.info(
            "Multilingual turn detection model initialized",
            extra={
                "model_repo": hf_repo,
                "model_filename": ONNX_FILENAME,
                "max_length": self._max_length
            }
        )

    def run(self, data: bytes) -> bytes | None:
        """Execute inference on the input data."""
        try:
            data_json = json.loads(data)
            sentence = data_json["sentence"]

            if not sentence:
                return json.dumps({"probability": 0.0}).encode()

            # Tokenize input
            inputs = self._tokenizer(
                sentence.strip(),
                truncation=True,
                max_length=self._max_length,
                return_tensors="np",
            )

            input_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            if "token_type_ids" in inputs:
                input_dict["token_type_ids"] = inputs["token_type_ids"]

            # Run inference
            outputs = self._session.run(None, input_dict)

            logits = outputs[0][0]

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)

            eou_probability = float(probabilities[1])

            logger.debug(
                "Multilingual turn detection inference completed",
                extra={
                    "probability": eou_probability,
                    "sentence_length": len(sentence),
                    "sentence": sentence
                },
            )

            return json.dumps({"probability": eou_probability}).encode()

        except Exception as e:
            logger.error(f"Error during turn detection inference: {e}")
            return json.dumps({"probability": 0.0, "error": str(e)}).encode()


class MultilingualModel(NamoModelBase):
    """
    Multilingual Namo Turn Detection model.

    This model works with 23+ languages including Arabic, Bengali, Chinese, Danish,
    Dutch, German, English, Finnish, French, Hindi, Indonesian, Italian, Japanese,
    Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish,
    Ukrainian, and Vietnamese.
    """

    def __init__(self, *, threshold: float = 0.7, **kwargs):
        """
        Initialize the multilingual model.

        Args:
            threshold: The threshold for end-of-utterance detection (0.0-1.0). Defaults to 0.5.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(threshold=threshold, **kwargs)

    def _inference_method(self) -> str:
        """Get the inference method for multilingual model."""
        return _NamoTurnRunnerMultilingual.INFERENCE_METHOD

    async def supports_language(self, language: str) -> bool:
        """
        Check if the model supports the given language.

        The multilingual model supports all languages.

        Args:
            language: Language code to check (e.g., 'en', 'fr', 'es')

        Returns:
            bool: Always True for multilingual model
        """
        return language in SUPPORTED_LANGUAGES


_InferenceRunner.register_runner(_NamoTurnRunnerMultilingual)
