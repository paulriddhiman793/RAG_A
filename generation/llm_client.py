"""
Groq Cloud LLM client.

Groq runs open-source models (Llama 3, Mixtral, Gemma) at extremely
low latency via custom LPU hardware. The API is OpenAI-compatible,
so the Groq SDK mirrors the openai client interface almost exactly.

Supported methods:
    complete(prompt, ...)          — text completion
    complete_json(prompt, ...)     — forces valid JSON output
    complete_vision(...)           — image + text (routes to llama-4-scout)

Rate limits (free tier, as of 2025):
    llama-3.3-70b-versatile : 6,000 tokens/min,  100 req/day
    llama-3.1-8b-instant    : 20,000 tokens/min, 14,400 req/day
    mixtral-8x7b-32768      : 5,000 tokens/min,  14,400 req/day
"""
from __future__ import annotations

import json
from typing import Any

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from utils.logger import logger


class LLMClient:
    """
    Groq Cloud client wired into the RAG pipeline.

    All pipeline components (query expander, answer generator,
    math parser, table summariser, figure alt-text) call this client.
    """

    def __init__(self):
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        logger.info(
            "Groq LLM client initialised | "
            f"text model: {self.model} | vision model: {settings.VISION_MODEL}"
        )

    # ── Text completion ───────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def complete(
        self,
        prompt: str,
        max_tokens: int | None = None,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """
        Single-turn text completion.

        Parameters
        ----------
        prompt      : user message
        max_tokens  : cap on output tokens (defaults to settings.MAX_TOKENS)
        system      : optional system prompt
        temperature : 0.1 for factual RAG tasks, higher for creative tasks
        """
        max_tokens = max_tokens or settings.MAX_TOKENS

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Groq completion failed: {e}")
            raise

    # ── JSON-mode completion ──────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Force JSON output using Groq's response_format feature.
        Returns parsed dict, or {} on parse failure.
        """
        json_system = (
            (system + "\n\n" if system else "")
            + "You MUST respond with valid JSON only. "
              "No markdown fences, no explanation. Raw JSON object only."
        )

        messages = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,                          # deterministic for JSON
                response_format={"type": "json_object"},  # Groq native JSON mode
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Groq JSON completion failed: {e}")
            raise

    # ── Vision completion ─────────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
    def complete_vision(
        self,
        text: str,
        image_base64: str,
        image_mime_type: str = "image/png",
        max_tokens: int = 300,
    ) -> str:
        """
        Vision completion — image + text prompt.

        Routes to the configured Groq vision model. Falls back to text-only if
        the call fails so the pipeline never hard-crashes on image chunks.
        """
        vision_model = settings.VISION_MODEL
        logger.info(f"Vision completion using model: {vision_model}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{image_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ]

        try:
            response = self._client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.warning(
                f"Groq vision call failed ({e}). "
                f"Falling back to text-only completion."
            )
            fallback_prompt = (
                f"Based on the figure caption below, write a brief description "
                f"suitable for a scientific search index.\n\nCaption / context: {text}"
            )
            return self.complete(fallback_prompt, max_tokens=max_tokens)

    # ── Convenience helpers ───────────────────────────────────────────

    def list_models(self) -> list[str]:
        """Return available model IDs from Groq."""
        try:
            models = self._client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.warning(f"Could not list Groq models: {e}")
            return []

    @property
    def model_info(self) -> dict:
        """Quick reference for the configured model."""
        notes = {
            "llama-3.3-70b-versatile": {
                "context": "128k tokens",
                "strength": "Best quality — recommended for RAG",
                "speed": "~300 tokens/s",
            },
            "llama-3.1-8b-instant": {
                "context": "128k tokens",
                "strength": "Fastest, lowest latency",
                "speed": "~750 tokens/s",
            },
            "mixtral-8x7b-32768": {
                "context": "32k tokens",
                "strength": "Strong reasoning, long context",
                "speed": "~500 tokens/s",
            },
            "gemma2-9b-it": {
                "context": "8k tokens",
                "strength": "Lightweight, good for summarisation",
                "speed": "~500 tokens/s",
            },
        }
        return notes.get(self.model, {"context": "unknown", "strength": "unknown"})
