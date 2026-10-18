"""LangChain chat-model factory with NVIDIA -> LM Studio fallback.

Usage:
    from rag01.llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")

Set LLM_PROVIDER in .env:
    auto       Try NVIDIA first; fall back to LM Studio on timeout/connection/
               rate-limit/server errors.
    nvidia     Use NVIDIA only.
    lm_studio  Use LM Studio only.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

load_dotenv()

DEFAULT_NVIDIA_MODEL = "deepseek-ai/deepseek-v4.1-flash"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_MODEL = "local-model"


def _make_nvidia_llm() -> ChatOpenAI:
    """Create the NVIDIA chat model using the OpenAI-compatible endpoint."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add it to .env or set "
            "LLM_PROVIDER=lm_studio."
        )

    return ChatOpenAI(
        model=os.getenv("NVIDIA_CHAT_MODEL", DEFAULT_NVIDIA_MODEL),
        base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL),
        api_key=api_key,
        temperature=0,
        timeout=float(os.getenv("NVIDIA_LLM_TIMEOUT", "30")),
        max_retries=0,
        max_tokens=512,
    )


def _make_lm_studio_llm() -> ChatOpenAI:
    """Create the local LM Studio chat model."""
    return ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL),
        base_url=os.getenv("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL),
        # LM Studio usually accepts any non-empty API key.
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        temperature=0,
        timeout=float(os.getenv("LM_STUDIO_LLM_TIMEOUT", "180")),
        max_retries=0,
        max_tokens=512,
    )


def get_llm() -> Any:
    """Return a LangChain chat model or a LangChain fallback runnable.

    In auto mode, LangChain's ``with_fallbacks`` handles supported transient
    provider failures. The returned object supports normal Runnable methods
    such as invoke/batch/stream, subject to the selected model endpoint.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    if provider == "nvidia":
        return _make_nvidia_llm()

    if provider == "lm_studio":
        return _make_lm_studio_llm()

    if provider != "auto":
        raise ValueError("LLM_PROVIDER must be one of: auto, nvidia, lm_studio")

    nvidia_llm = _make_nvidia_llm()
    lm_studio_llm = _make_lm_studio_llm()

    recoverable_errors = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    return nvidia_llm.with_fallbacks(
        [lm_studio_llm],
        exceptions_to_handle=recoverable_errors,
    )
