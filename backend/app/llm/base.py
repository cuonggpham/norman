"""
Base LLM class - abstract interface for LLM implementations.

Provides common functionality and defines the contract that all
LLM providers must follow.
"""

from abc import ABC, abstractmethod
from typing import Any

from .prompts import LEGAL_ASSISTANT_SYSTEM, RAG_USER_TEMPLATE


class BaseLLM(ABC):
    """
    Base class for LLM implementations.
    
    Subclasses must implement the `generate` method.
    The `generate_with_context` method provides a default RAG implementation.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of {"role": "...", "content": "..."}
            **kwargs: Provider-specific options
            
        Returns:
            Generated text
        """
        pass
    
    def generate_with_context(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate response with RAG context.
        
        Default implementation builds messages and calls generate().
        Override for custom behavior.
        
        Args:
            query: User's question
            context: List of retrieved chunks
            system_prompt: Optional system prompt override
            
        Returns:
            Generated answer
        """
        messages = self._build_rag_messages(query, context, system_prompt)
        return self.generate(messages)
    
    def _build_rag_messages(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build chat messages for RAG.
        
        Args:
            query: User's question
            context: List of retrieved chunks
            system_prompt: Optional system prompt override
            
        Returns:
            List of chat messages
        """
        # Use default system prompt if not provided
        system = system_prompt or LEGAL_ASSISTANT_SYSTEM
        
        # Format context into numbered sections
        context_text = "\n\n".join([
            f"【文書 {i+1}】\n{chunk}"
            for i, chunk in enumerate(context)
        ])
        
        # Build user message with context
        user_content = RAG_USER_TEMPLATE.format(
            context=context_text,
            query=query,
        )
        
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
