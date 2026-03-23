from .logger import logger
from .language_detector import detect_language, is_multilingual
from .coref_resolver import resolve_coreferences, resolve_batch

__all__ = [
    "logger",
    "detect_language",
    "is_multilingual",
    "resolve_coreferences",
    "resolve_batch",
]
