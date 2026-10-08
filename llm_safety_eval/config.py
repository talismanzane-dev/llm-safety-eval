"""Configuration loading for llm-safety-eval.

API keys are read exclusively from environment variables (optionally loaded
from a ``.env`` file via ``python-dotenv``). They are never written to disk,
logged, or included in result files.

Recognised environment variables
--------------------------------
API keys:
    ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``DEEPSEEK_API_KEY``,
    ``GOOGLE_API_KEY``

Model overrides (optional):
    ``CLAUDE_MODEL``, ``OPENAI_MODEL``, ``DEEPSEEK_MODEL``, ``GEMINI_MODEL``,
    ``DEEPSEEK_BASE_URL`` (default ``https://api.deepseek.com``)

Per-provider sampling overrides (optional, use ``none`` to omit the parameter):
    ``CLAUDE_TEMPERATURE``, ``OPENAI_TEMPERATURE``, ``DEEPSEEK_TEMPERATURE``,
    ``GEMINI_TEMPERATURE``, ``LOCAL_TEMPERATURE``

Local OpenAI-compatible server (optional, e.g. Ollama, vLLM, llama.cpp):
    ``LOCAL_MODEL_NAME`` (enables the local model when set),
    ``LOCAL_MODEL_BASE_URL`` (default ``http://localhost:11434/v1``),
    ``LOCAL_MODEL_API_KEY`` (optional)

Global settings (optional):
    ``EVAL_MAX_TOKENS`` (default 1024), ``EVAL_TIMEOUT_SECONDS`` (default 60),
    ``EVAL_MAX_RETRIES`` (default 2), ``EVAL_SYSTEM_PROMPT``,
    ``EVAL_RESULTS_DIR`` (default ``data/results``)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:  # python-dotenv is a declared dependency, but keep config importable without it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only in minimal environments
    load_dotenv = None  # type: ignore[assignment]


class Provider(str, Enum):
    """Supported API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    LOCAL = "local"
    MOCK = "mock"


class ModelConfig(BaseModel):
    """Configuration for a single model endpoint.

    Attributes:
        name: Short, human-readable identifier used in results and reports
            (e.g. ``"claude"``). Must be unique within a run.
        provider: Which API family the model is served through.
        model_id: The provider-specific model identifier.
        api_key_env: Name of the environment variable holding the API key.
            ``None`` for endpoints that need no key (local servers, mocks).
        base_url: Optional override for the API endpoint.
        max_tokens: Maximum number of tokens to generate per response.
        temperature: Sampling temperature. ``None`` omits the parameter and
            uses the provider default (required by some reasoning models).
        timeout_seconds: Per-request timeout.
        max_retries: SDK-level retries for transient errors (rate limits, 5xx).
        system_prompt: Optional system prompt sent with every request.
    """

    name: str
    provider: Provider
    model_id: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = Field(default=1024, gt=0)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    system_prompt: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        """Return the API key from the environment, or ``None`` if unset.

        The key is resolved lazily on every access so it is never stored on
        the config object (and therefore never serialised with it).
        """
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None

    @property
    def is_available(self) -> bool:
        """Whether the model has the credentials it needs to be called."""
        return self.api_key_env is None or self.api_key is not None


class EvalConfig(BaseModel):
    """Top-level configuration: every known model plus global settings."""

    models: dict[str, ModelConfig]
    results_dir: Path = Path("data/results")

    def get(self, name: str) -> ModelConfig:
        """Return the config for ``name``.

        Raises:
            KeyError: If no model with that name is configured.
        """
        try:
            return self.models[name]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "<none>"
            raise KeyError(f"Unknown model '{name}'. Configured models: {known}") from None

    def available_models(self) -> list[ModelConfig]:
        """Return the models whose required API keys are present."""
        return [m for m in self.models.values() if m.is_available]

    def select(self, names: Optional[list[str]] = None) -> list[ModelConfig]:
        """Select models by name, or every available model if ``names`` is empty.

        Raises:
            KeyError: If a requested model is not configured.
            ValueError: If a requested model is missing its API key.
        """
        if not names:
            return self.available_models()
        selected = [self.get(n) for n in names]
        missing = [m for m in selected if not m.is_available]
        if missing:
            details = ", ".join(f"{m.name} (set {m.api_key_env})" for m in missing)
            raise ValueError(f"Missing API keys for: {details}")
        return selected


def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env_str(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = _env_str(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be a number, got {raw!r}") from exc


def _env_temperature(key: str, default: Optional[float]) -> Optional[float]:
    """Read a temperature; the literal ``none`` means "omit the parameter"."""
    raw = _env_str(key)
    if raw is None:
        return default
    if raw.lower() in {"none", "null", "default"}:
        return None
    return _env_float(key, default if default is not None else 0.0)


def load_config(env_file: Optional[str | Path] = None, override: bool = False) -> EvalConfig:
    """Build an :class:`EvalConfig` from environment variables.

    Args:
        env_file: Optional path to a ``.env`` file. If omitted, a ``.env`` in
            the current working directory (or a parent) is loaded if present.
        override: If true, values in the ``.env`` file override variables that
            are already set in the process environment.

    Returns:
        An :class:`EvalConfig` describing Claude, GPT, DeepSeek, Gemini and
        (when ``LOCAL_MODEL_NAME`` is set) a local OpenAI-compatible model.
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_file, override=override)

    max_tokens = _env_int("EVAL_MAX_TOKENS", 1024)
    timeout = _env_float("EVAL_TIMEOUT_SECONDS", 60.0)
    retries = _env_int("EVAL_MAX_RETRIES", 2)
    system_prompt = _env_str("EVAL_SYSTEM_PROMPT")

    common = {
        "timeout_seconds": timeout,
        "max_retries": retries,
        "system_prompt": system_prompt,
    }

    models: dict[str, ModelConfig] = {
        "claude": ModelConfig(
            name="claude",
            provider=Provider.ANTHROPIC,
            model_id=_env_str("CLAUDE_MODEL", "claude-sonnet-5-5"),
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=max_tokens,
            # Current Claude models don't accept a temperature; set
            # CLAUDE_TEMPERATURE only when targeting an older model.
            temperature=_env_temperature("CLAUDE_TEMPERATURE", None),
            **common,
        ),
        "gpt": ModelConfig(
            name="gpt",
            provider=Provider.OPENAI,
            model_id=_env_str("OPENAI_MODEL", "gpt-5"),
            api_key_env="OPENAI_API_KEY",
            # Reasoning models spend completion tokens on hidden reasoning, so
            # give them headroom to produce a visible answer.
            max_tokens=max(max_tokens, 4096),
            # GPT-5-family models only accept the default temperature.
            temperature=_env_temperature("OPENAI_TEMPERATURE", None),
            **common,
        ),
        "deepseek": ModelConfig(
            name="deepseek",
            provider=Provider.DEEPSEEK,
            model_id=_env_str("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key_env="DEEPSEEK_API_KEY",
            base_url=_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            max_tokens=max_tokens,
            temperature=_env_temperature("DEEPSEEK_TEMPERATURE", 0.0),
            **common,
        ),
        "gemini": ModelConfig(
            name="gemini",
            provider=Provider.GEMINI,
            model_id=_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key_env="GOOGLE_API_KEY",
            max_tokens=max_tokens,
            temperature=_env_temperature("GEMINI_TEMPERATURE", 0.0),
            **common,
        ),
    }

    local_model = _env_str("LOCAL_MODEL_NAME")
    if local_model:
        models["local"] = ModelConfig(
            name="local",
            provider=Provider.LOCAL,
            model_id=local_model,
            # Only require a key if the user configured one.
            api_key_env="LOCAL_MODEL_API_KEY" if _env_str("LOCAL_MODEL_API_KEY") else None,
            base_url=_env_str("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1"),
            max_tokens=max_tokens,
            temperature=_env_temperature("LOCAL_TEMPERATURE", 0.0),
            **common,
        )

    return EvalConfig(
        models=models,
        results_dir=Path(_env_str("EVAL_RESULTS_DIR", "data/results")),
    )
