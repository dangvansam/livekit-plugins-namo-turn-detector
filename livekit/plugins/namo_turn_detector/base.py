# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import time
from abc import ABC, abstractmethod

from livekit.agents import ChatContext, ChatMessage, llm
from livekit.agents.ipc.inference_executor import InferenceExecutor
from livekit.agents.job import get_job_context

from .log import logger


class NamoModelBase(ABC):
    """
    Base class for Namo Turn Detection models.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        inference_executor: InferenceExecutor | None = None,
    ) -> None:
        """
        Initialize the base model.

        Args:
            threshold: The threshold for end-of-utterance detection (0.0-1.0). Defaults to 0.5.
            inference_executor: Optional custom inference executor. If None, uses job context executor.
        """
        self._threshold = threshold
        self._executor = inference_executor or get_job_context().inference_executor
        self._languages: dict[str, dict] = {}  # Cache for language-specific thresholds

    @property
    def threshold(self) -> float:
        """Get the current detection threshold."""
        return self._threshold

    @property
    def provider(self) -> str:
        return "namo"
    
    @property
    def model(self) -> str:
        return "namo"

    @abstractmethod
    def _inference_method(self) -> str:
        """
        Get the inference method name for this model.
        """
        ...

    async def unlikely_threshold(self, language: str | None) -> float | None:
        """
        Get the threshold for language.
        """
        return self._threshold

    def _get_last_user_message(self, chat_ctx: ChatContext) -> str:
        """
        Extract the last user message from chat context.

        Args:
            chat_ctx: The chat context to analyze

        Returns:
            str: The last user message content
        """
        user_messages = [
            item
            for item in chat_ctx.items
            if isinstance(item, ChatMessage) and item.role == "user"
        ]

        if not user_messages:
            return ""

        last_message = user_messages[-1]

        # Handle different content types
        if hasattr(last_message, "text_content"):
            return last_message.text_content.strip()

        content = last_message.content

        if isinstance(content, list):
            text_content = " ".join(
                [c.text if hasattr(c, "text") else str(c) for c in content]
            )
        else:
            text_content = str(content)

        return text_content.strip()

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = 10.0,
    ) -> float:
        """
        Predict the probability of end-of-turn for the given chat context.

        This method extracts the last user message and runs inference to determine
        if the user has finished their turn.

        Args:
            chat_ctx: The chat context to analyze
            timeout: Maximum time to wait for inference (seconds). Defaults to 3.0.

        Returns:
            float: Probability score between 0.0 and 1.0, where higher values
                   indicate higher confidence that the turn has ended.

        Raises:
            asyncio.TimeoutError: If inference takes longer than timeout
            RuntimeError: If inference execution fails

        Example:
            ```python
            probability = await model.predict_end_of_turn(chat_ctx)
            if probability >= model.threshold:
                print("End of turn detected!")
            ```
        """
        start_time = time.time()

        try:
            # Extract the last user message
            sentence = self._get_last_user_message(chat_ctx)

            if not sentence:
                logger.debug("No user message found in chat context")
                return 0.0

            # Prepare data for inference
            json_data = json.dumps({"sentence": sentence}).encode()

            # Execute inference with timeout
            result = await asyncio.wait_for(
                self._executor.do_inference(self._inference_method(), json_data),
                timeout=timeout,
            )

            # Parse result
            result_json = json.loads(result.decode())
            probability = result_json.get("probability", 0.0)
            probability = round(probability, 3)

            duration = time.time() - start_time
            logger.debug(
                f"Predict EOT Finished: {probability >= self.threshold}",
                extra={
                    "probability": probability,
                    "threshold": self.threshold,
                    "sentence": sentence,
                    "duration": duration
                }
            )
            return probability

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Turn detection inference timeout",
                extra={"duration": duration, "timeout": timeout},
            )
            raise

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                f"Error during turn detection: {e}",
                extra={"duration": duration, "error": str(e)},
            )
            return 0.0