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

from livekit.agents import Plugin

from .log import logger
from .version import __version__

__all__ = [
    "multilingual",
    "language_specific",
    "vi_model",
    "en_model",
    "zh_model",
    "__version__",
]


class NamoTurnDetectorPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        from .download_model import download_model

        # Download multilingual model
        download_model()

        # Download language-specific models (en, vi, zh)
        for lang in ["en", "vi", "zh"]:
            download_model(language=lang)


Plugin.register_plugin(NamoTurnDetectorPlugin())


def __getattr__(name: str):
    if name == "multilingual":
        from . import multilingual

        return multilingual
    elif name == "language_specific":
        from . import language_specific

        return language_specific
    elif name == "vi_model":
        from . import vi_model

        return vi_model
    elif name == "en_model":
        from . import en_model

        return en_model
    elif name == "zh_model":
        from . import zh_model

        return zh_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
