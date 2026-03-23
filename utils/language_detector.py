from lingua import Language, LanguageDetectorBuilder
from langdetect import detect as langdetect_detect, DetectorFactory
from utils.logger import logger

# Make langdetect deterministic
DetectorFactory.seed = 42

# Build a fast lingua detector for the most common languages
_DETECTOR = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)

# Map lingua Language enum to BCP-47 codes
_LINGUA_TO_BCP47: dict[Language, str] = {
    Language.ENGLISH: "en",
    Language.HINDI: "hi",
    Language.GERMAN: "de",
    Language.FRENCH: "fr",
    Language.SPANISH: "es",
    Language.PORTUGUESE: "pt",
    Language.CHINESE: "zh",
    Language.JAPANESE: "ja",
    Language.ARABIC: "ar",
    Language.RUSSIAN: "ru",
    Language.KOREAN: "ko",
    Language.ITALIAN: "it",
    Language.DUTCH: "nl",
    Language.POLISH: "pl",
    Language.TURKISH: "tr",
}


def detect_language(text: str) -> str:
    """
    Detect the dominant language of a text string.
    Returns a BCP-47 language code (e.g. 'en', 'hi', 'de').
    Falls back to 'en' on failure.
    """
    if not text or len(text.strip()) < 20:
        return "en"

    try:
        lang = _DETECTOR.detect_language_of(text)
        if lang and lang in _LINGUA_TO_BCP47:
            return _LINGUA_TO_BCP47[lang]
    except Exception:
        pass

    # Fallback to langdetect
    try:
        return langdetect_detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


def is_multilingual(texts: list[str]) -> bool:
    """Return True if the corpus contains more than one language."""
    langs = {detect_language(t) for t in texts if t.strip()}
    return len(langs) > 1