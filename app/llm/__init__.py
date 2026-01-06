"""LLM Layer - Providers and utilities for LLM integration."""

from .base import BaseLLM
from .openai_provider import OpenAIProvider
from .prompts import LEGAL_ASSISTANT_SYSTEM, RAG_USER_TEMPLATE
from .query_translator import QueryTranslator

__all__ = [
    "BaseLLM",
    "OpenAIProvider",
    "QueryTranslator",
    "LEGAL_ASSISTANT_SYSTEM",
    "RAG_USER_TEMPLATE",
]

