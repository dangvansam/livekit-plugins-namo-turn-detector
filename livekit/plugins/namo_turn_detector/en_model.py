import json

import numpy as np
from huggingface_hub import hf_hub_download
from livekit.agents.inference_runner import _InferenceRunner

from .base import NamoModelBase
from .log import logger
from .model import ONNX_FILENAME

HF_REPO = "videosdk-live/Namo-Turn-Detector-v1-English"
LANGUAGE_CODE = "en"


class _NamoTurnRunnerEnglish(_InferenceRunner):
    """Inference runner for English turn detection model."""

    INFERENCE_METHOD = "namo_turn_detection_en"

    def initialize(self) -> None:
        """Load English model."""
        import onnxruntime as ort
        from transformers import DistilBertTokenizer

        logger.info(f"Loading English turn detection model from {HF_REPO}")

        self._tokenizer = DistilBertTokenizer.from_pretrained(
            HF_REPO, truncation_side="left"
        )
        self._max_length = 512

        model_path = hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILENAME)

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        logger.info(
            "English model loaded",
            extra={
                "language": LANGUAGE_CODE,
                "model_repo": HF_REPO,
                "model_filename": ONNX_FILENAME,
                "max_length": self._max_length,
            }
        )

    def run(self, data: bytes) -> bytes | None:
        """Execute inference on English text."""
        try:
            data_json = json.loads(data)
            sentence = data_json["sentence"]

            if not sentence:
                return json.dumps({"probability": 0.0}).encode()

            # Tokenize
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
                f"English turn detection completed",
                extra={
                    "probability": eou_probability,
                    "sentence_length": len(sentence),
                    "sentence": sentence,
                },
            )

            return json.dumps({"probability": eou_probability}).encode()

        except Exception as e:
            logger.error(f"Error during English turn detection: {e}")
            return json.dumps({"probability": 0.0, "error": str(e)}).encode()


_InferenceRunner.register_runner(_NamoTurnRunnerEnglish)


class EnglishModel(NamoModelBase):
    """
    English-specific Namo Turn Detection model.

    This model loads only the English model file for better memory efficiency.
    """

    def __init__(self, *, threshold: float = 0.7, **kwargs):
        """
        Initialize English model.

        Args:
            threshold: Detection threshold (0.0-1.0). Defaults to 0.7.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(threshold=threshold, **kwargs)

    async def supports_language(self, language: str) -> bool:
        """
        Check if the model supports the given language.

        Args:
            language: Language code to check (e.g., 'en', 'fr', 'es')

        Returns:
            bool: True if this model is configured for the given language
        """
        return language.lower() == LANGUAGE_CODE

    @property
    def language(self) -> str:
        """Get language code."""
        return LANGUAGE_CODE

    @property
    def model(self) -> str:
        return f"namo-{LANGUAGE_CODE}"

    def _inference_method(self) -> str:
        """Get inference method name."""
        return _NamoTurnRunnerEnglish.INFERENCE_METHOD
