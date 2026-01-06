"""
Query Translator for Cross-lingual RAG.

Translates Vietnamese queries to Japanese for better semantic search
against Japanese legal documents.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider (to avoid circular imports)."""
    def generate(self, messages: list[dict[str, str]], **kwargs) -> str: ...


# Translation prompt
TRANSLATION_SYSTEM = """Bạn là dịch giả chuyên ngành pháp luật Nhật Bản.
Dịch câu hỏi pháp luật từ tiếng Việt sang tiếng Nhật.

Quy tắc:
1. Giữ nguyên các thuật ngữ pháp lý đã có tiếng Nhật (例: 労働基準法)
2. Dịch ngắn gọn, tập trung vào keywords pháp lý
3. Chỉ trả về bản dịch tiếng Nhật, không giải thích
4. Nếu input đã là tiếng Nhật, trả về nguyên bản"""


class QueryTranslator:
    """
    Translates Vietnamese queries to Japanese for semantic search.
    
    Uses LLM to translate while preserving legal terminology.
    
    Example:
        translator = QueryTranslator(llm=OpenAIProvider(...))
        ja_query = translator.translate("Quy định về thời gian làm việc")
        # Returns: "労働時間に関する規定"
    """
    
    def __init__(self, llm: LLMProvider):
        """
        Initialize translator.
        
        Args:
            llm: LLM provider for translation
        """
        self._llm = llm
    
    def translate(self, query: str) -> str:
        """
        Translate Vietnamese query to Japanese.
        
        Args:
            query: Vietnamese query string
            
        Returns:
            Japanese translation for embedding/search
        """
        # Skip if query is likely already Japanese
        if self._is_japanese(query):
            return query
        
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM},
            {"role": "user", "content": query},
        ]
        
        # Use lower temperature for consistent translation
        translated = self._llm.generate(messages, temperature=0.1, max_tokens=256)
        
        return translated.strip()
    
    def _is_japanese(self, text: str) -> bool:
        """
        Check if text is primarily Japanese.
        
        Simple heuristic: contains Hiragana, Katakana, or Kanji.
        """
        for char in text:
            # Hiragana: U+3040-U+309F
            # Katakana: U+30A0-U+30FF
            # CJK Unified: U+4E00-U+9FFF
            if '\u3040' <= char <= '\u309f':  # Hiragana
                return True
            if '\u30a0' <= char <= '\u30ff':  # Katakana
                return True
            if '\u4e00' <= char <= '\u9fff':  # Kanji
                # Could be Chinese, but in this context likely Japanese
                return True
        return False
