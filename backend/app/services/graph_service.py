"""
Graph Search Service for GraphRAG.
Provides methods to query the Neo4j knowledge graph for legal entities.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import re

from app.db.neo4j_client import get_neo4j_client, Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    """Result from graph search."""
    law_id: str
    law_title: str
    article_num: str
    article_title: Optional[str]
    article_caption: Optional[str]
    chunk_id: Optional[str]
    relevance: float
    path: List[str]  # Highlight path
    source: str = "graph"


class GraphService:
    """Service for graph-based retrieval operations."""
    
    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize graph service.
        
        Args:
            client: Optional Neo4j client (uses singleton if not provided)
        """
        self.client = client or get_neo4j_client()
    
    def find_article(self, law_title: str, article_num: str) -> Optional[GraphResult]:
        """
        Find a specific article by law name and article number.
        
        Args:
            law_title: Law title (partial match allowed)
            article_num: Article number (e.g., "32", "32-2")
        
        Returns:
            GraphResult if found, None otherwise
        """
        query = """
        MATCH (l:Law)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_ARTICLE]->(a:Article)
        WHERE l.title CONTAINS $law_title AND a.num = $article_num
        OPTIONAL MATCH (a)-[:HAS_PARAGRAPH]->(p:Paragraph)
        RETURN l.law_id as law_id, l.title as law_title, 
               a.num as article_num, a.title as article_title, a.caption as caption,
               collect(p.chunk_id)[0] as chunk_id
        LIMIT 1
        """
        
        try:
            results = self.client.run_query(query, {
                "law_title": law_title,
                "article_num": article_num
            })
            
            if results:
                r = results[0]
                return GraphResult(
                    law_id=r["law_id"],
                    law_title=r["law_title"],
                    article_num=r["article_num"],
                    article_title=r.get("article_title"),
                    article_caption=r.get("caption"),
                    chunk_id=r.get("chunk_id"),
                    relevance=1.0,
                    path=[r["law_title"], f"第{r['article_num']}条"]
                )
        except Exception as e:
            logger.warning(f"Error finding article: {e}")
        
        return None
    
    def find_article_by_num(self, law_id: str, article_num: str) -> Optional[GraphResult]:
        """
        Find article by law_id and article number.
        
        Args:
            law_id: Law ID (e.g., "322AC0000000049")
            article_num: Article number
        
        Returns:
            GraphResult if found
        """
        query = """
        MATCH (l:Law {law_id: $law_id})-[:HAS_CHAPTER]->(:Chapter)-[:HAS_ARTICLE]->(a:Article {num: $article_num})
        OPTIONAL MATCH (a)-[:HAS_PARAGRAPH]->(p:Paragraph)
        RETURN l.law_id as law_id, l.title as law_title,
               a.num as article_num, a.title as article_title, a.caption as caption,
               collect(p.chunk_id)[0] as chunk_id
        LIMIT 1
        """
        
        try:
            results = self.client.run_query(query, {
                "law_id": law_id,
                "article_num": article_num
            })
            
            if results:
                r = results[0]
                return GraphResult(
                    law_id=r["law_id"],
                    law_title=r["law_title"],
                    article_num=r["article_num"],
                    article_title=r.get("article_title"),
                    article_caption=r.get("caption"),
                    chunk_id=r.get("chunk_id"),
                    relevance=1.0,
                    path=[r["law_title"], f"第{r['article_num']}条"]
                )
        except Exception as e:
            logger.warning(f"Error finding article by num: {e}")
        
        return None
    
    def find_related_articles(self, law_id: str, article_num: str, 
                              depth: int = 2, limit: int = 10) -> List[GraphResult]:
        """
        Find articles connected by REFERENCES relationships.
        
        Args:
            law_id: Law ID
            article_num: Starting article number
            depth: Maximum relationship depth (1-3)
            limit: Maximum results
        
        Returns:
            List of related articles
        """
        # Use variable-length path pattern
        query = """
        MATCH (start:Article {law_id: $law_id, num: $article_num})
        MATCH path = (start)-[:REFERENCES*1..2]-(related:Article)
        WHERE related.law_id = $law_id AND related.num <> $article_num
        MATCH (l:Law {law_id: related.law_id})
        OPTIONAL MATCH (related)-[:HAS_PARAGRAPH]->(p:Paragraph)
        RETURN DISTINCT l.law_id as law_id, l.title as law_title,
               related.num as article_num, related.title as article_title,
               related.caption as caption,
               collect(DISTINCT p.chunk_id)[0] as chunk_id,
               length(path) as distance
        ORDER BY distance
        LIMIT $limit
        """
        
        results = []
        try:
            data = self.client.run_query(query, {
                "law_id": law_id,
                "article_num": article_num,
                "limit": limit
            })
            
            for r in data:
                results.append(GraphResult(
                    law_id=r["law_id"],
                    law_title=r["law_title"],
                    article_num=r["article_num"],
                    article_title=r.get("article_title"),
                    article_caption=r.get("caption"),
                    chunk_id=r.get("chunk_id"),
                    relevance=0.95 ** r.get("distance", 1),  # Exponential decay: 0.95, 0.90, 0.86, ...
                    path=[r["law_title"], f"第{r['article_num']}条"]
                ))
        except Exception as e:
            logger.warning(f"Error finding related articles: {e}")
        
        return results
    
    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[GraphResult]:
        """
        Search articles by keyword in title/caption.
        
        Args:
            keyword: Search keyword
            limit: Maximum results
        
        Returns:
            List of matching articles
        """
        query = """
        MATCH (l:Law)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_ARTICLE]->(a:Article)
        WHERE a.title CONTAINS $keyword OR a.caption CONTAINS $keyword
              OR l.title CONTAINS $keyword
        OPTIONAL MATCH (a)-[:HAS_PARAGRAPH]->(p:Paragraph)
        RETURN DISTINCT l.law_id as law_id, l.title as law_title,
               a.num as article_num, a.title as article_title, a.caption as caption,
               collect(DISTINCT p.chunk_id)[0] as chunk_id
        LIMIT $limit
        """
        
        results = []
        try:
            data = self.client.run_query(query, {
                "keyword": keyword,
                "limit": limit
            })
            
            for r in data:
                results.append(GraphResult(
                    law_id=r["law_id"],
                    law_title=r["law_title"],
                    article_num=r["article_num"],
                    article_title=r.get("article_title"),
                    article_caption=r.get("caption"),
                    chunk_id=r.get("chunk_id"),
                    relevance=0.8,
                    path=[r["law_title"], f"第{r['article_num']}条"]
                ))
        except Exception as e:
            logger.warning(f"Error searching by keyword: {e}")
        
        return results
    
    def get_law_structure(self, law_id: str) -> Dict[str, Any]:
        """
        Get hierarchical structure of a law.
        
        Args:
            law_id: Law ID
        
        Returns:
            Dictionary with law structure
        """
        query = """
        MATCH (l:Law {law_id: $law_id})-[:HAS_CHAPTER]->(c:Chapter)
        OPTIONAL MATCH (c)-[:HAS_ARTICLE]->(a:Article)
        RETURN l.title as law_title, c.num as chapter_num, c.title as chapter_title,
               collect({num: a.num, title: a.title, caption: a.caption}) as articles
        ORDER BY c.num
        """
        
        try:
            results = self.client.run_query(query, {"law_id": law_id})
            
            if not results:
                return {}
            
            return {
                "law_id": law_id,
                "law_title": results[0]["law_title"],
                "chapters": [
                    {
                        "num": r["chapter_num"],
                        "title": r["chapter_title"],
                        "articles": [a for a in r["articles"] if a.get("num")]
                    }
                    for r in results
                ]
            }
        except Exception as e:
            logger.warning(f"Error getting law structure: {e}")
            return {}
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get current graph statistics."""
        stats = {}
        stats["nodes"] = self.client.get_node_counts()
        stats["relationships"] = self.client.get_relationship_counts()
        return stats


# Singleton instance
_graph_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    """Get singleton GraphService instance."""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
