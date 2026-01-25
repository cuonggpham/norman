"""
Query Translator for Cross-lingual RAG.

Translates Vietnamese queries to Japanese for better semantic search
against Japanese legal documents.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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


# Query expansion prompt
QUERY_EXPANSION_SYSTEM = """Bạn là chuyên gia pháp luật lao động Nhật Bản.
Phân tích câu hỏi pháp lý và trả về JSON với các thông tin sau:

1. translated: Bản dịch tiếng Nhật của câu hỏi
2. keywords: 3-5 keywords pháp lý tiếng Nhật liên quan (重要!)
3. related_terms: 2-3 thuật ngữ liên quan (có thể là số điều, tên luật)
4. search_queries: 2-3 câu query tìm kiếm khác nhau (tiếng Nhật)

Trả về CHÍNH XÁC format JSON sau, không có text khác:
{"translated": "...", "keywords": [...], "related_terms": [...], "search_queries": [...]}

Ví dụ cho câu hỏi "Thời gian làm việc tối đa một tuần là bao nhiêu giờ?":
{"translated": "週の最大労働時間は何時間ですか？", "keywords": ["労働時間", "法定労働時間", "週40時間", "一週間"], "related_terms": ["第三十二条", "労働基準法"], "search_queries": ["法定労働時間の上限", "週の労働時間制限", "労働基準法の労働時間規定"]}"""


@dataclass
class QueryExpansion:
    """Result of query expansion."""
    original: str
    translated: str
    keywords: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


class QueryTranslator:
    """
    Translates Vietnamese queries to Japanese for semantic search.
    
    Uses LLM to translate while preserving legal terminology.
    Now with query expansion for better retrieval.
    
    Example:
        translator = QueryTranslator(llm=OpenAIProvider(...))
        ja_query = translator.translate("Quy định về thời gian làm việc")
        # Returns: "労働時間に関する規定"
        
        expansion = translator.expand("Quy định về sa thải nhân viên")
        # Returns QueryExpansion with keywords, related_terms, search_queries
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
    
    def expand(self, query: str) -> QueryExpansion:
        """
        Expand query with keywords and multiple search queries.
        
        Args:
            query: Vietnamese query string
            
        Returns:
            QueryExpansion with translated query, keywords, and search queries
        """
        # If already Japanese, still extract keywords
        messages = [
            {"role": "system", "content": QUERY_EXPANSION_SYSTEM},
            {"role": "user", "content": query},
        ]
        
        try:
            response = self._llm.generate(messages, temperature=0.2, max_tokens=512)
            
            # Parse JSON response
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            data = json.loads(response)
            
            return QueryExpansion(
                original=query,
                translated=data.get("translated", query),  # ⚡ OPTIMIZATION: Use query directly, no fallback LLM call
                keywords=data.get("keywords", []),
                related_terms=data.get("related_terms", []),
                search_queries=data.get("search_queries", []),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Query expansion failed, using original query: {e}")
            # ⚡ OPTIMIZATION: Return original query directly without calling translate()
            # This saves 1 LLM call. Most queries are already Japanese after initial detection.
            return QueryExpansion(
                original=query,
                translated=query,
                keywords=[],
                related_terms=[],
                search_queries=[],
            )
    
    def get_all_search_texts(self, query: str) -> list[str]:
        """
        Get all texts to use for search (for multi-query retrieval).
        
        Returns list of:
        1. Original translated query
        2. Additional search queries from expansion
        
        Args:
            query: Vietnamese query string
            
        Returns:
            List of Japanese search texts
        """
        expansion = self.expand(query)
        
        texts = [expansion.translated]
        texts.extend(expansion.search_queries)
        
        # Add keyword combinations
        if expansion.keywords:
            keyword_query = " ".join(expansion.keywords[:3])
            texts.append(keyword_query)
        
        return texts
    
    def _is_japanese(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text is primarily Japanese based on character ratio.
        
        Uses ratio-based detection instead of single character check to handle
        mixed Vietnamese-Japanese queries like "Thuế quà tặng (贈与税)..." which
        should be translated (primarily Vietnamese with Japanese terms).
        
        Args:
            text: Input text to check
            threshold: Minimum ratio of Japanese chars to consider as Japanese (default 50%)
        
        Returns:
            True if Japanese characters exceed threshold ratio
        """
        if not text:
            return False
        
        jp_chars = 0
        total_chars = 0
        
        for char in text:
            # Skip whitespace and common punctuation
            if char.isspace() or char in '()（）「」、。？！.,?!':
                continue
            
            total_chars += 1
            
            # Japanese character ranges
            if '\u3040' <= char <= '\u309f':  # Hiragana
                jp_chars += 1
            elif '\u30a0' <= char <= '\u30ff':  # Katakana
                jp_chars += 1
            elif '\u4e00' <= char <= '\u9fff':  # Kanji/CJK
                jp_chars += 1
        
        if total_chars == 0:
            return False
        
        ratio = jp_chars / total_chars
        return ratio >= threshold

