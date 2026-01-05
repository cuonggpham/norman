"""LLM Layer - Providers and utilities for LLM integration."""

from .base import BaseLLM
from .openai_provider import OpenAIProvider
from .prompts import LEGAL_ASSISTANT_SYSTEM, RAG_USER_TEMPLATE

__all__ = [
    "BaseLLM",
    "OpenAIProvider",
    "LEGAL_ASSISTANT_SYSTEM",
    "RAG_USER_TEMPLATE",
]
