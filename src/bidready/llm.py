"""Model providers. mock does not call a model. ollama and openai read keys from the environment."""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from bidready.config import Settings


class LLMClient(Protocol):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    def complete_json(self, *, system: str, user: str) -> dict: ...


class MockLLM:
    provider = "mock"
    model = "heuristic-v1"
    prompt_tokens = 0
    completion_tokens = 0

    def complete_json(self, *, system: str, user: str) -> dict:
        raise RuntimeError("mock provider does not call a model; the graph uses the deterministic agents")


class _ChatLLM:
    prompt_tokens = 0
    completion_tokens = 0

    def complete_json(self, *, system: str, user: str) -> dict:
        text = self._chat(system=system, user=user)
        try:
            return _parse_json(text)
        except json.JSONDecodeError:
            text = self._chat(
                system=system,
                user=user + "\n\nYour previous reply was not JSON. Return only one JSON object.",
            )
            return _parse_json(text)

    def _chat(self, *, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(_ChatLLM):
    provider = "openai"

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _chat(self, *, system: str, user: str) -> str:
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return payload["choices"][0]["message"]["content"]


class OllamaLLM(_ChatLLM):
    """Local models via Ollama. Suggested model: qwen2.5:7b-instruct. Override with LLM_MODEL."""

    provider = "ollama"

    def __init__(self, *, base_url: str, model: str) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")

    def _chat(self, *, system: str, user: str) -> str:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        # Ollama reports token counts on the response when eval_count is present.
        # Record them only when the server sends the fields. Do not estimate.
        if "prompt_eval_count" in payload:
            self.prompt_tokens += int(payload.get("prompt_eval_count") or 0)
        if "eval_count" in payload:
            self.completion_tokens += int(payload.get("eval_count") or 0)
        return payload["message"]["content"]


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "mock":
        llm = MockLLM()
        if settings.prompt_version == "v2":
            llm.model = "heuristic-v2"
        return llm
    if provider == "openai":
        return OpenAICompatibleLLM(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model or "gpt-4o-mini",
        )
    if provider == "ollama":
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.llm_model or "qwen2.5:7b-instruct",
        )
    raise RuntimeError(f"unknown LLM_PROVIDER: {provider}")


def _parse_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected an object", text, 0)
    return parsed
