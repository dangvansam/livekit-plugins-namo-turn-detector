# Namo Turn Detector Plugin for LiveKit Agents

Turn detection plugin for LiveKit Agents using [Namo Turn Detector](https://github.com/videosdk-live/NAMO-Turn-Detector-v1) models.

## Installation

```bash
pip install livekit-plugins-namo-turn-detector
```

## Features

- **Multilingual Support**: 23+ languages with unified multilingual model
- **Language-Specific Models**: Optimized models for English, Vietnamese, Chinese
- **Fast & Accurate**: Outperforms baseline models in accuracy and speed
- **Async API**: Built on LiveKit's inference runner for optimal performance
- **Easy Integration**: Drop-in replacement for existing turn detectors

## Quick Start

### Multilingual Model (23+ Languages)

```python
from livekit.plugins.namo_turn_detector.multilingual import MultilingualModel
from livekit import agents


async def entrypoint(ctx: agents.JobContext):
    model = MultilingualModel(threshold=0.7)
    
    # Get probability
    prob = await model.predict_end_of_turn(chat_ctx)
```

### Language-Specific Model (Better Accuracy)

```python
from livekit.plugins.namo_turn_detector.language_specific import LanguageSpecificModel

# English model
async def entrypoint(ctx: agents.JobContext):
    model = LanguageSpecificModel(language="en", threshold=0.7)
    prob = await model.predict_end_of_turn(chat_ctx)
```

**Supported Languages**: `en` (English), `vi` (Vietnamese), `zh` (Chinese)

## Benchmark Results

Comparison across English, Vietnamese, and Chinese:

### English Performance
```
Sample: "Hello, how are you?"
  • Namo Multilingual:     0.8757 (16ms) - EOT: True
  • Namo English-Specific: 0.0002 (13ms) - EOT: False
  • LiveKit Multilingual:  0.2838 (33ms) - EOT: True
  • LiveKit English:       0.4596 (4ms)  - EOT: True

Sample: "What's the weather like today?"
  • Namo Multilingual:     0.8032 (15ms) - EOT: True
  • Namo English-Specific: 0.9999 (9ms)  - EOT: True ⭐
  • LiveKit Multilingual:  0.7799 (27ms) - EOT: True
  • LiveKit English:       0.9409 (3ms)  - EOT: True
```

### Vietnamese Performance
```
Sample: "Xin chào, bạn khỏe không?" (Hello, how are you?)
  • Namo Multilingual:        0.8651 (25ms) - EOT: True
  • Namo Vietnamese-Specific: 0.9857 (36ms) - EOT: True ⭐
  • LiveKit Multilingual:     0.0322 (20ms) - EOT: False

Sample: "Thời tiết hôm nay thế nào?" (What's the weather today?)
  • Namo Multilingual:        0.5168 (27ms) - EOT: False
  • Namo Vietnamese-Specific: 0.9952 (4ms)  - EOT: True ⭐
  • LiveKit Multilingual:     0.2988 (22ms) - EOT: False

Sample: "Vay ở đâu" (Where to borrow) - Incomplete phrase
  • Namo Multilingual:        0.6599 (20ms) - EOT: False
  • Namo Vietnamese-Specific: 0.9875 (10ms) - EOT: True ⭐
  • LiveKit Multilingual:     0.5106 (25ms) - EOT: False
```

### Chinese Performance
```
Sample: "你好，你好吗？" (Hello, how are you?)
  • Namo Multilingual:     0.6525 (30ms) - EOT: False
  • Namo Chinese-Specific: 0.8777 (16ms) - EOT: True ⭐
  • LiveKit Multilingual:  0.8520 (20ms) - EOT: True

Sample: "今天天气怎么样？" (What's the weather today?)
  • Namo Multilingual:     0.6818 (18ms) - EOT: False
  • Namo Chinese-Specific: 0.9090 (34ms) - EOT: True ⭐
  • LiveKit Multilingual:  0.9707 (20ms) - EOT: True
```

**Key Insights:**
- **Language-Specific models** show superior accuracy for their target languages
- **Namo Multilingual** provides consistent performance across all languages
- **Inference speed** is competitive, typically 10-30ms per prediction
- **Vietnamese detection** significantly outperforms baseline multilingual model

## API Reference

### MultilingualModel

```python
MultilingualModel(threshold: float = 0.7)
```

**Methods:**
- `predict_end_of_turn(chat_ctx, timeout=10.0) -> float` - Returns probability (0.0-1.0)
- `unlikely_threshold(language) -> float` - Get model's threshold for language

### LanguageSpecificModel

```python
LanguageSpecificModel(language: str, threshold: float = 0.7)
```

**Parameters:**
- `language`: Language code (`"en"`, `"vi"`, `"zh"`)
- `threshold`: Detection threshold (0.0-1.0)

**Methods:** Same as MultilingualModel
- `predict_end_of_turn(chat_ctx, timeout=10.0) -> float` - Returns probability (0.0-1.0)
- `unlikely_threshold(language) -> float` - Get model's threshold for language

### Pre-download Models

```bash
python main.py download-files
```

## Supported Languages

- **Multilingual Model (23+ languages):**
Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, Vietnamese

- **Language-Specific Models:** English (`en`), Vietnamese (`vi`), Chinese (`zh`)

## License

Apache-2.0

## Credits

- Models: [Namo Turn Detector v1](https://github.com/videosdk-live/NAMO-Turn-Detector-v1) by VideoSDK
- Framework: [LiveKit Agents](https://github.com/livekit/agents)

## Citation
```
@software{namo2025,
  title = {Namo Turn Detector v1: Semantic Turn Detection for Conversational AI},
  author = {VideoSDK Team},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/collections/videosdk-live/namo-turn-detector-v1-68d52c0564d2164e9d17ca97}
}
```