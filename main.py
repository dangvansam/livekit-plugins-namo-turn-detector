from livekit import agents
from livekit.agents import ChatContext
from livekit.plugins.namo_turn_detector.multilingual import MultilingualModel as NamoMultilingual
from livekit.plugins.namo_turn_detector.language_specific import LanguageSpecificModel as NamoLanguageSpecific
from livekit.plugins.turn_detector.english import EnglishModel as LiveKitEnglish
from livekit.plugins.turn_detector.multilingual import MultilingualModel as LiveKitMultilingual


async def entrypoint(ctx: agents.JobContext):

    print("\n" + "=" * 80)
    print("Turn Detection Model Comparison Test")
    print("=" * 80)

    # Test samples in different languages
    test_samples = {
        "English": [
            "Hello, how are you?",
            "What's the weather like today?",
            "I need help with my computer",
        ],
        "Vietnamese": [
            "Xin chào, bạn khỏe không?",
            "Thời tiết hôm nay thế nào?",
            "Tôi cần giúp đỡ về máy tính",
            "Em đang cần",
            "Mình muốn",
            "Vay ở đâu",
            "Có nhé",
            "Ok",
            "Anh đang bận nhé, gọi lại sau cho anh",
        ],
        "Chinese": [
            "你好，你好吗？",
            "今天天气怎么样？",
            "我需要电脑方面的帮助",
        ],
    }

    # Language codes for models
    lang_codes = {
        "English": "en",
        "Vietnamese": "vi",
        "Chinese": "zh",
    }

    # Initialize all models
    print("\n📦 Initializing models...")
    namo_multilingual = NamoMultilingual()
    livekit_multilingual = LiveKitMultilingual()
    livekit_english = LiveKitEnglish()

    # Language-specific models for each language
    namo_language_specific = {
        "en": NamoLanguageSpecific(language="en"),
        "vi": NamoLanguageSpecific(language="vi"),
        "zh": NamoLanguageSpecific(language="zh"),
    }

    print("✓ All models initialized\n")

    # Test each language
    for language, messages in test_samples.items():
        lang_code = lang_codes[language]
        print("\n" + "─" * 80)
        print(f"🌍 Testing: {language} ({lang_code.upper()})")
        print("─" * 80)

        for idx, message in enumerate(messages, 1):
            import time

            print("=" * 80)
            print(f"\n📝 Sample {idx}: \"{message}\"")

            # Create chat context
            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content=message)

            # Test Namo Multilingual
            start = time.time()
            prob_namo_multi = await namo_multilingual.predict_end_of_turn(chat_ctx)
            time_namo_multi = (time.time() - start) * 1000  # Convert to ms
            threshold_namo_multi = await namo_multilingual.unlikely_threshold(lang_code)
            eot_namo_multi = prob_namo_multi >= threshold_namo_multi if threshold_namo_multi else False

            # Test Namo Language-Specific
            start = time.time()
            prob_namo_specific = await namo_language_specific[lang_code].predict_end_of_turn(chat_ctx)
            time_namo_specific = (time.time() - start) * 1000  # Convert to ms
            threshold_namo_specific = await namo_language_specific[lang_code].unlikely_threshold(lang_code)
            eot_namo_specific = prob_namo_specific >= threshold_namo_specific if threshold_namo_specific else False

            # Test LiveKit Multilingual
            start = time.time()
            prob_livekit_multi = await livekit_multilingual.predict_end_of_turn(chat_ctx)
            time_livekit_multi = (time.time() - start) * 1000  # Convert to ms
            threshold_livekit_multi = await livekit_multilingual.unlikely_threshold(lang_code)
            eot_livekit_multi = prob_livekit_multi >= threshold_livekit_multi if threshold_livekit_multi else False

            # Test LiveKit English (only for English samples)
            if language == "English":
                start = time.time()
                prob_livekit_en = await livekit_english.predict_end_of_turn(chat_ctx)
                time_livekit_en = (time.time() - start) * 1000  # Convert to ms
                threshold_livekit_en = await livekit_english.unlikely_threshold(lang_code)
                eot_livekit_en = prob_livekit_en >= threshold_livekit_en if threshold_livekit_en else False
            else:
                prob_livekit_en = None
                time_livekit_en = None
                eot_livekit_en = None

            # Display results with timing and EOT decision
            print(f"  • Namo Multilingual:        {prob_namo_multi:.4f} ({time_namo_multi:.2f}ms) - EOT: {eot_namo_multi}")
            print(f"  • Namo {language}-Specific: {prob_namo_specific:.4f} ({time_namo_specific:.2f}ms) - EOT: {eot_namo_specific}")
            print(f"  • LiveKit Multilingual:     {prob_livekit_multi:.4f} ({time_livekit_multi:.2f}ms) - EOT: {eot_livekit_multi}")
            if prob_livekit_en is not None:
                print(f"  • LiveKit English:          {prob_livekit_en:.4f} ({time_livekit_en:.2f}ms) - EOT: {eot_livekit_en}")
            print("=" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("\nModel Summary:")
    print("  • Namo Multilingual:       Supports 23+ languages")
    print("  • Namo Language-Specific:  Optimized per language (en, vi, zh)")
    print("  • LiveKit Multilingual:    Baseline multilingual model")
    print("  • LiveKit English:         English-only model")
    print("=" * 80)

    ctx.shutdown()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from livekit.agents import WorkerOptions, cli

    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
