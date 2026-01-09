"""
Query Analyzer for RAG Pipeline.

Analyzes queries to detect category, keywords, and intent
for better retrieval filtering.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Category mapping - Vietnamese/Japanese terms to category
CATEGORY_KEYWORDS = {
    "労働": [
        # Vietnamese
        "thời gian làm việc", "giờ làm việc", "làm thêm giờ", "tăng ca", "nghỉ phép",
        "phép năm", "sa thải", "giải thuê", "nghỉ việc", "hợp đồng lao động",
        "nghỉ giữa ca", "ngày nghỉ", "lương", "tiền công", "làm đêm", "ca đêm",
        "điều kiện lao động", "lao động",
        # Japanese
        "労働時間", "残業", "時間外労働", "有給休暇", "年次休暇", "解雇", "退職",
        "労働契約", "休憩時間", "休日", "賃金", "深夜労働", "労働条件",
    ],
    "社会保険": [
        # Vietnamese
        "bảo hiểm", "bảo hiểm xã hội", "bảo hiểm y tế", "hưu trí", "thất nghiệp",
        # Japanese
        "社会保険", "健康保険", "年金", "失業", "保険",
    ],
    "国税": [
        # Vietnamese
        "thuế", "thuế thu nhập", "thuế doanh nghiệp", "khai thuế",
        # Japanese
        "税", "所得税", "法人税", "申告",
    ],
    "災害補償": [
        # Vietnamese
        "tai nạn lao động", "bồi thường", "chấn thương", "bệnh nghề nghiệp",
        # Japanese
        "災害補償", "労災", "業務上", "負傷", "疾病", "療養",
    ],
}


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    detected_category: str | None = None
    detected_keywords: list[str] = field(default_factory=list)
    suggested_filters: dict = field(default_factory=dict)
    confidence: float = 0.0


class QueryAnalyzer:
    """
    Analyzes queries to detect category and suggest filters.
    
    Uses keyword matching for fast, deterministic results.
    No LLM calls required.
    
    Example:
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Quy định về sa thải nhân viên")
        # Returns QueryAnalysis with category="労働"
    """
    
    def __init__(self, category_keywords: dict[str, list[str]] | None = None):
        """
        Initialize analyzer.
        
        Args:
            category_keywords: Custom category-to-keywords mapping
        """
        self.category_keywords = category_keywords or CATEGORY_KEYWORDS
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query to detect category and suggest filters.
        
        Args:
            query: User query (Vietnamese or Japanese)
            
        Returns:
            QueryAnalysis with detected category and suggestions
        """
        query_lower = query.lower()
        
        # Count keyword matches per category
        category_scores = {}
        matched_keywords = {}
        
        for category, keywords in self.category_keywords.items():
            matches = []
            for kw in keywords:
                if kw.lower() in query_lower:
                    matches.append(kw)
            
            if matches:
                category_scores[category] = len(matches)
                matched_keywords[category] = matches
        
        # Find best category
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            total_matches = sum(category_scores.values())
            confidence = best_score / total_matches if total_matches > 0 else 0.0
            
            # Only return category if confidence is high enough
            if confidence >= 0.5:
                return QueryAnalysis(
                    original_query=query,
                    detected_category=best_category,
                    detected_keywords=matched_keywords.get(best_category, []),
                    suggested_filters={"category": best_category},
                    confidence=confidence,
                )
        
        # No clear category detected
        return QueryAnalysis(
            original_query=query,
            detected_category=None,
            detected_keywords=[],
            suggested_filters={},
            confidence=0.0,
        )
    
    def get_suggested_filters(self, query: str) -> dict:
        """
        Get suggested filters for a query.
        
        Convenience method that returns just the filters dict.
        
        Args:
            query: User query
            
        Returns:
            Dict of suggested filters (may be empty)
        """
        analysis = self.analyze(query)
        return analysis.suggested_filters


# Singleton instance
_analyzer = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get singleton query analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer()
    return _analyzer
