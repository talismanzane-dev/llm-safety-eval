"""Model abstraction layer.

Every supported provider is wrapped behind the same interface::

    client = create_client(config)
    response = client.generate("Some prompt")   # -> ModelResponse

``generate`` never raises for API failures. Errors are captured on the
returned :class:`ModelResponse` (``response.error``) so a single failing call
cannot abort a long evaluation run. Programming errors such as a missing SDK
are raised eagerly when the client is constructed.

Provider SDKs are imported lazily, so you only need to install the SDKs for
the providers you actually use.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, Field

from .config import ModelConfig, Provider


class ModelResponse(BaseModel):
    """Normalised result of a single generation call.

    Attributes:
        model_name: The configured short name of the model (e.g. ``"claude"``).
        provider: The provider that served the request.
        model_id: The provider-specific model identifier.
        prompt: The prompt that was sent.
        text: The visible text of the response (empty on error or block).
        finish_reason: The provider's raw stop/finish reason, if any.
        provider_refusal: True when the provider explicitly signalled a
            refusal or safety block (e.g. Anthropic ``stop_reason="refusal"``,
            OpenAI ``message.refusal``, Gemini ``SAFETY`` blocks). This is a
            strong signal; text-based heuristics are applied separately by
            the harness.
        refusal_detail: Provider-supplied refusal explanation, if any.
        latency_seconds: Wall-clock time for the call, including retries.
        input_tokens: Prompt tokens reported by the provider, if available.
        output_tokens: Completion tokens reported by the provider, if available.
        error: Error message if the call failed, otherwise ``None``.
    """

    model_name: str
    provider: str
    model_id: str
    prompt: str
    text: str = ""
    finish_reason: Optional[str] = None
    provider_refusal: bool = False
    refusal_detail: Optional[str] = None
    latency_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if the call completed without an error."""
        return self.error is None


class LLMClient(ABC):
    """Abstract base class for all model clients.

    Subclasses implement :meth:`_generate`, which performs the API call and
    returns a partially populated :class:`ModelResponse`. The public
    :meth:`generate` method adds timing and error handling.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        """The configured short name of the model."""
        return self.config.name

    def generate(self, prompt: str) -> ModelResponse:
        """Send ``prompt`` to the model and return a normalised response.

        This method never raises for API or network failures; inspect
        ``response.error`` instead.
        """
        start = time.perf_counter()
        try:
            response = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: record, don't crash
            response = self._empty_response(prompt, error=f"{type(exc).__name__}: {exc}")
        response.latency_seconds = time.perf_counter() - start
        return response

    @abstractmethod
    def _generate(self, prompt: str) -> ModelResponse:
        """Perform the provider-specific API call."""

    def _empty_response(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Build a response pre-filled with this client's identity."""
        return ModelResponse(
            model_name=self.config.name,
            provider=self.config.provider.value,
            model_id=self.config.model_id,
            prompt=prompt,
            **kwargs,
        )

    def _require_api_key(self) -> str:
        key = self.config.api_key
        if not key:
            raise ValueError(
                f"Model '{self.config.name}' requires the {self.config.api_key_env} "
                "environment variable to be set."
            )
        return key

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.config.name!r}, model_id={self.config.model_id!r})"


class AnthropicClient(LLMClient):
    """Client for Claude models via the Anthropic Messages API."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install the Anthropic SDK: pip install anthropic") from exc
        self._client = anthropic.Anthropic(
            api_key=self._require_api_key(),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def _generate(self, prompt: str) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.system_prompt:
            kwargs["system"] = self.config.system_prompt
        if self.config.temperature is not None:
            # Current Claude models and SDK releases no longer expose a
            # temperature parameter; pass it through raw for older models.
            kwargs["extra_body"] = {"temperature": self.config.temperature}

        message = self._client.messages.create(**kwargs)

        text = "".join(
            getattr(block, "text", "") for block in message.content if getattr(block, "type", None) == "text"
        )
        stop_reason = getattr(message, "stop_reason", None)
        usage = getattr(message, "usage", None)
        return self._empty_response(
            prompt,
            text=text,
            finish_reason=stop_reason,
            provider_refusal=stop_reason == "refusal",
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


class OpenAIClient(LLMClient):
    """Client for OpenAI chat models (and any OpenAI-compatible SDK endpoint)."""

    #: Name of the token-limit parameter. OpenAI's newer models require
    #: ``max_completion_tokens``; most compatible servers use ``max_tokens``.
    max_tokens_param = "max_completion_tokens"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK: pip install openai") from exc
        self._client = openai.OpenAI(
            api_key=self._require_api_key(),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def _generate(self, prompt: str) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            self.max_tokens_param: self.config.max_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        completion = self._client.chat.completions.create(**kwargs)

        if not completion.choices:
            return self._empty_response(prompt, error="Response contained no choices")
        choice = completion.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(completion, "usage", None)
        return self._empty_response(
            prompt,
            text=message.content or "",
            finish_reason=finish_reason,
            provider_refusal=bool(refusal) or finish_reason == "content_filter",
            refusal_detail=refusal or None,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


class DeepSeekClient(OpenAIClient):
    """Client for DeepSeek models via their OpenAI-compatible endpoint."""

    max_tokens_param = "max_tokens"

    def __init__(self, config: ModelConfig) -> None:
        if not config.base_url:
            config = config.model_copy(update={"base_url": "https://api.deepseek.com"})
        super().__init__(config)


class GeminiClient(LLMClient):
    """Client for Gemini models via the ``google-generativeai`` SDK."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Install the Gemini SDK: pip install google-generativeai") from exc
        # Note: genai.configure() sets process-global state. All Gemini clients
        # in one process therefore share a single API key.
        genai.configure(api_key=self._require_api_key())

        generation_config: dict[str, Any] = {"max_output_tokens": config.max_tokens}
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature

        self._model = genai.GenerativeModel(
            model_name=config.model_id,
            generation_config=generation_config,
            system_instruction=config.system_prompt or None,
        )

    def _generate(self, prompt: str) -> ModelResponse:
        result = self._model.generate_content(
            prompt,
            request_options={"timeout": self.config.timeout_seconds},
        )

        usage = getattr(result, "usage_metadata", None)
        token_kwargs = {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        }

        # Prompt-level block: no candidates are returned at all.
        feedback = getattr(result, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason and _enum_name(block_reason) not in {"BLOCK_REASON_UNSPECIFIED", "0"}:
            return self._empty_response(
                prompt,
                finish_reason=f"PROMPT_BLOCKED:{_enum_name(block_reason)}",
                provider_refusal=True,
                refusal_detail=_enum_name(block_reason),
                **token_kwargs,
            )

        candidates = list(getattr(result, "candidates", None) or [])
        if not candidates:
            return self._empty_response(prompt, error="Response contained no candidates", **token_kwargs)

        candidate = candidates[0]
        finish = _enum_name(getattr(candidate, "finish_reason", None))
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(part, "text", "") or "" for part in parts)
        blocked = finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
        return self._empty_response(
            prompt,
            text=text,
            finish_reason=finish,
            provider_refusal=blocked,
            refusal_detail=finish if blocked else None,
            **token_kwargs,
        )


class LocalOpenAICompatibleClient(LLMClient):
    """Client for local models served over an OpenAI-compatible HTTP API.

    Works with Ollama (``http://localhost:11434/v1``), vLLM, llama.cpp's
    server, LM Studio and similar. Uses plain ``requests`` so no vendor SDK is
    required.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if not config.base_url:
            raise ValueError(f"Local model '{config.name}' requires a base_url")
        self._url = config.base_url.rstrip("/") + "/chat/completions"
        self._session = requests.Session()
        key = config.api_key
        if key:
            self._session.headers["Authorization"] = f"Bearer {key}"

    def _generate(self, prompt: str) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._session.post(self._url, json=payload, timeout=self.config.timeout_seconds)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp)
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status >= 500 or status == 429
                if not retryable or attempt == self.config.max_retries:
                    raise
                time.sleep(2**attempt)
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError(f"Request failed: {last_error}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return self._empty_response(prompt, error="Response contained no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        finish_reason = choice.get("finish_reason")
        refusal = message.get("refusal")
        return self._empty_response(
            prompt,
            text=message.get("content") or "",
            finish_reason=finish_reason,
            provider_refusal=bool(refusal) or finish_reason == "content_filter",
            refusal_detail=refusal or None,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class MockClient(LLMClient):
    """Offline client for dry runs and tests. Makes no network calls.

    Args:
        config: Model configuration. Only ``name`` and ``model_id`` are used.
        responder: Function mapping a prompt to response text. Defaults to a
            deterministic responder that refuses prompts containing any of
            ``refuse_keywords`` and answers everything else.
        refuse_keywords: Case-insensitive substrings that trigger a refusal
            in the default responder.
    """

    DEFAULT_REFUSAL = "I'm sorry, but I can't help with that request."

    def __init__(
        self,
        config: ModelConfig,
        responder: Optional[Callable[[str], str]] = None,
        refuse_keywords: Optional[list[str]] = None,
    ) -> None:
        super().__init__(config)
        self._keywords = [k.lower() for k in (refuse_keywords or [])]
        self._responder = responder or self._default_responder

    def _default_responder(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(k in lowered for k in self._keywords):
            return self.DEFAULT_REFUSAL
        return f"[mock:{self.config.name}] Here is a helpful answer to: {prompt[:80]}"

    def _generate(self, prompt: str) -> ModelResponse:
        text = self._responder(prompt)
        return self._empty_response(prompt, text=text, finish_reason="stop")


_CLIENTS: dict[Provider, type[LLMClient]] = {
    Provider.ANTHROPIC: AnthropicClient,
    Provider.OPENAI: OpenAIClient,
    Provider.DEEPSEEK: DeepSeekClient,
    Provider.GEMINI: GeminiClient,
    Provider.LOCAL: LocalOpenAICompatibleClient,
    Provider.MOCK: MockClient,
}


def create_client(config: ModelConfig) -> LLMClient:
    """Instantiate the right :class:`LLMClient` subclass for ``config``.

    Raises:
        ValueError: If the provider is unknown or a required API key is missing.
        ImportError: If the provider's SDK is not installed.
    """
    try:
        cls = _CLIENTS[config.provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {config.provider}") from None
    return cls(config)


def create_clients(configs: list[ModelConfig]) -> list[LLMClient]:
    """Instantiate a client for each config, preserving order."""
    return [create_client(c) for c in configs]


def _enum_name(value: Any) -> str:
    """Return a stable string name for a protobuf/Python enum or raw value."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)
