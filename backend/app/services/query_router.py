"""
Query Router for Hybrid RAG.
Analyzes queries to determine the best retrieval strategy.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for routing."""
    SEMANTIC = "semantic"           # General question → Vector search
    ENTITY_LOOKUP = "entity_lookup" # "第32条 là gì?" → Graph lookup
    MULTI_HOP = "multi_hop"         # "Điều liên quan đến..." → Graph traversal
    HYBRID = "hybrid"               # Combination of both


@dataclass
class RoutedQuery:
    """Result of query routing."""
    original_query: str
    query_type: QueryType
    entities: List[Tuple[str, str]]  # List of (entity, entity_type)
    use_graph: bool
    use_vector: bool


class QueryRouter:
    """
    Route queries to appropriate retrieval method.
    
    Analyzes query text to extract legal entities and determine
    whether to use graph search, vector search, or both.
    """
    
    # Patterns for entity extraction
    ENTITY_PATTERNS = [
        # 労働基準法第32条 → (労働基準法, 32)
        (r'([ぁ-んァ-ン一-龯]+法)第(\d+)条', 'law_article'),
        # 第32条 (standalone article)
        (r'第(\d+)条(?:の(\d+))?', 'article_only'),
        # Law names
        (r'([ぁ-んァ-ン一-龯]+法)', 'law_name'),
    ]
    
    # Keywords indicating relationship queries
    RELATIONSHIP_KEYWORDS = [
        "liên quan", "related", "tham chiếu", "references",
        "kết nối", "connected", "điều khác", "các điều",
        "quy định tại", "theo điều", "dựa trên"
    ]
    
    # Keywords indicating entity lookup
    LOOKUP_KEYWORDS = [
        "là gì", "nói gì", "quy định gì", "what is",
        "điều", "khoản", "mục", "chương"
    ]
    
    def route(self, query: str) -> RoutedQuery:
        """
        Analyze query and determine routing strategy.
        
        Args:
            query: User query text
        
        Returns:
            RoutedQuery with routing information
        """
        entities = self._extract_entities(query)
        is_relationship = self._is_relationship_query(query)
        is_lookup = self._is_lookup_query(query)
        
        # Determine query type
        if entities and is_lookup and not is_relationship:
            # Direct entity lookup: "第32条 nói gì?"
            query_type = QueryType.ENTITY_LOOKUP
            use_graph = True
            use_vector = False
            
        elif entities and is_relationship:
            # Multi-hop relationship query
            query_type = QueryType.MULTI_HOP
            use_graph = True
            use_vector = True  # Also get semantic context
            
        elif entities:
            # Has entities but not clear intent → hybrid
            query_type = QueryType.HYBRID
            use_graph = True
            use_vector = True
            
        else:
            # No entities → pure semantic search
            query_type = QueryType.SEMANTIC
            use_graph = False
            use_vector = True
        
        logger.debug(f"Query routed: type={query_type.value}, entities={entities}")
        
        return RoutedQuery(
            original_query=query,
            query_type=query_type,
            entities=entities,
            use_graph=use_graph,
            use_vector=use_vector
        )
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract legal entities from text.
        
        Returns:
            List of (entity_value, entity_type) tuples
        """
        entities = []
        
        for pattern, entity_type in self.ENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                if entity_type == 'law_article':
                    # Full reference: 労働基準法第32条
                    law_name = match.group(1)
                    article_num = match.group(2)
                    entities.append((f"{law_name}第{article_num}条", 'law_article'))
                    
                elif entity_type == 'article_only':
                    # Just article: 第32条
                    article_num = match.group(1)
                    sub_num = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                    if sub_num:
                        entities.append((f"第{article_num}条の{sub_num}", 'article'))
                    else:
                        entities.append((f"第{article_num}条", 'article'))
                        
                elif entity_type == 'law_name':
                    # Just law name: 労働基準法
                    # Only add if not already captured as part of law_article
                    law_name = match.group(1)
                    if not any(law_name in e[0] for e in entities):
                        entities.append((law_name, 'law'))
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        
        return unique
    
    def _is_relationship_query(self, query: str) -> bool:
        """Check if query asks about relationships between entities."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.RELATIONSHIP_KEYWORDS)
    
    def _is_lookup_query(self, query: str) -> bool:
        """Check if query is a direct entity lookup."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.LOOKUP_KEYWORDS)
    
    def parse_law_article_reference(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse a law+article reference like "労働基準法第32条".
        
        Returns:
            Tuple of (law_name, article_num) or None
        """
        match = re.search(r'([ぁ-んァ-ン一-龯]+法)第(\d+)条', text)
        if match:
            return (match.group(1), match.group(2))
        return None
    
    def parse_article_reference(self, text: str) -> Optional[str]:
        """
        Parse an article reference like "第32条".
        
        Returns:
            Article number or None
        """
        match = re.search(r'第(\d+)条(?:の(\d+))?', text)
        if match:
            num = match.group(1)
            sub = match.group(2)
            return f"{num}-{sub}" if sub else num
        return None


# Singleton
_query_router: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """Get singleton QueryRouter instance."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
    return _query_router
