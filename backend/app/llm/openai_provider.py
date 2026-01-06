"""
OpenAI LLM Provider.

Implementation of BaseLLM using OpenAI's Chat API.
"""

from typing import Any

from openai import OpenAI

from .base import BaseLLM


class OpenAIProvider(BaseLLM):
    """
    OpenAI SDK implementation of LLM provider.
    
    Example:
        provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
        response = provider.generate([
            {"role": "user", "content": "Hello!"}
        ])
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.3  # Lower for more consistent legal responses
    DEFAULT_MAX_TOKENS = 2048
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o-mini, gpt-4o, etc.)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate response using OpenAI Chat API.
        
        Args:
            messages: Chat messages
            **kwargs: Override default params (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content or ""
    
    def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ):
        """
        Generate response with streaming.
        
        Yields:
            Text chunks as they arrive
        """
        stream = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
