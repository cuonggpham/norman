#!/usr/bin/env python3
"""
Graph Builder for Japanese Law Documents.
Builds Neo4j knowledge graph from processed JSON law files.

Creates nodes: Law, Chapter, Article, Paragraph
Creates relationships: HAS_CHAPTER, HAS_ARTICLE, HAS_PARAGRAPH

Run: python -m scripts.graph_builder
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.neo4j import get_neo4j_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class BuildStats:
    """Statistics for graph building."""
    laws: int = 0
    chapters: int = 0
    articles: int = 0
    paragraphs: int = 0
    errors: int = 0


def create_law_node(client, law_data: Dict[str, Any]) -> Optional[str]:
    """
    Create Law node from law data.
    
    Returns:
        law_id if successful, None otherwise
    """
    law_info = law_data.get("law_info", {})
    revision = law_data.get("revision_info", {})
    
    law_id = law_info.get("law_id", "")
    if not law_id:
        return None
    
    query = """
    MERGE (l:Law {law_id: $law_id})
    SET l.title = $title,
        l.title_kana = $title_kana,
        l.abbrev = $abbrev,
        l.category = $category,
        l.law_type = $law_type,
        l.law_num = $law_num,
        l.promulgation_date = $promulgation_date
    RETURN l.law_id as law_id
    """
    
    try:
        result = client.run_query(query, {
            "law_id": law_id,
            "title": law_info.get("law_title", ""),
            "title_kana": law_info.get("law_title_kana", ""),
            "abbrev": law_info.get("abbreviation", ""),
            "category": law_info.get("category", ""),
            "law_type": law_info.get("law_type", ""),
            "law_num": law_info.get("law_num", ""),
            "promulgation_date": revision.get("promulgation_date", ""),
        })
        return result[0]["law_id"] if result else None
    except Exception as e:
        logger.error(f"Error creating Law node {law_id}: {e}")
        return None


def create_chapter_nodes(client, law_id: str, chapters: List[Dict]) -> int:
    """
    Create Chapter nodes with HAS_CHAPTER relationships.
    
    Returns:
        Number of chapters created
    """
    query = """
    MATCH (l:Law {law_id: $law_id})
    MERGE (c:Chapter {law_id: $law_id, num: $num})
    SET c.title = $title
    MERGE (l)-[:HAS_CHAPTER {order: $order}]->(c)
    """
    
    count = 0
    for i, chapter in enumerate(chapters):
        try:
            client.run_write(query, {
                "law_id": law_id,
                "num": chapter.get("num", str(i + 1)),
                "title": chapter.get("title", ""),
                "order": i + 1
            })
            count += 1
        except Exception as e:
            logger.warning(f"Error creating chapter: {e}")
    
    return count


def create_article_nodes(client, law_id: str, chapter_num: str, 
                         articles: List[Dict]) -> int:
    """
    Create Article nodes with HAS_ARTICLE relationships.
    
    Returns:
        Number of articles created
    """
    query = """
    MATCH (c:Chapter {law_id: $law_id, num: $chapter_num})
    MERGE (a:Article {law_id: $law_id, num: $num})
    SET a.title = $title,
        a.caption = $caption,
        a.chapter_num = $chapter_num
    MERGE (c)-[:HAS_ARTICLE {order: $order}]->(a)
    """
    
    count = 0
    for i, article in enumerate(articles):
        try:
            client.run_write(query, {
                "law_id": law_id,
                "chapter_num": chapter_num,
                "num": article.get("num", str(i + 1)),
                "title": article.get("title", ""),
                "caption": article.get("caption", ""),
                "order": i + 1
            })
            count += 1
        except Exception as e:
            logger.warning(f"Error creating article: {e}")
    
    return count


def create_paragraph_nodes(client, law_id: str, article_num: str,
                           paragraphs: List[Dict], source_type: str = "main") -> int:
    """
    Create Paragraph nodes linked to existing chunks in vector store.
    
    Returns:
        Number of paragraphs created
    """
    query = """
    MATCH (a:Article {law_id: $law_id, num: $article_num})
    MERGE (p:Paragraph {chunk_id: $chunk_id})
    SET p.num = $num,
        p.law_id = $law_id,
        p.article_num = $article_num
    MERGE (a)-[:HAS_PARAGRAPH {order: $order}]->(p)
    """
    
    count = 0
    for i, para in enumerate(paragraphs):
        para_num = para.get("num", str(i + 1))
        # Generate chunk_id matching the chunker.py format
        chunk_id = f"{law_id}_{source_type}_{article_num}_{para_num}"
        
        try:
            client.run_write(query, {
                "law_id": law_id,
                "article_num": article_num,
                "chunk_id": chunk_id,
                "num": para_num,
                "order": i + 1
            })
            count += 1
        except Exception as e:
            logger.warning(f"Error creating paragraph: {e}")
    
    return count


def process_law_file(client, law_data: Dict[str, Any]) -> BuildStats:
    """
    Process a single law file and build graph nodes.
    
    Args:
        client: Neo4j client
        law_data: Parsed JSON law data
        
    Returns:
        BuildStats with counts
    """
    stats = BuildStats()
    
    # 1. Create Law node
    law_id = create_law_node(client, law_data)
    if not law_id:
        stats.errors += 1
        return stats
    stats.laws = 1
    
    # 2. Get main provision
    law_body = law_data.get("law_full_text", {}).get("law_body", {})
    main_provision = law_body.get("main_provision", {})
    
    # 3. Process chapters (if exists)
    chapters = main_provision.get("chapters", [])
    if chapters:
        stats.chapters = create_chapter_nodes(client, law_id, chapters)
        
        for chapter in chapters:
            chapter_num = chapter.get("num", "")
            articles = chapter.get("articles", [])
            stats.articles += create_article_nodes(client, law_id, chapter_num, articles)
            
            for article in articles:
                paragraphs = article.get("paragraphs", [])
                stats.paragraphs += create_paragraph_nodes(
                    client, law_id, article.get("num", ""), paragraphs
                )
    
    # 4. Process direct articles (without chapters)
    direct_articles = main_provision.get("articles", [])
    if direct_articles:
        # Create virtual chapter "0" for direct articles
        client.run_write("""
            MATCH (l:Law {law_id: $law_id})
            MERGE (c:Chapter {law_id: $law_id, num: "0"})
            SET c.title = "本則"
            MERGE (l)-[:HAS_CHAPTER {order: 0}]->(c)
        """, {"law_id": law_id})
        stats.chapters += 1
        
        stats.articles += create_article_nodes(client, law_id, "0", direct_articles)
        
        for article in direct_articles:
            paragraphs = article.get("paragraphs", [])
            stats.paragraphs += create_paragraph_nodes(
                client, law_id, article.get("num", ""), paragraphs
            )
    
    # 5. Process supplementary provisions
    suppl_provisions = law_body.get("suppl_provisions", [])
    for i, suppl in enumerate(suppl_provisions):
        suppl_articles = suppl.get("articles", [])
        for article in suppl_articles:
            paragraphs = article.get("paragraphs", [])
            stats.paragraphs += create_paragraph_nodes(
                client, law_id, article.get("num", ""), paragraphs,
                source_type="suppl"
            )
    
    return stats


def build_full_graph(limit: Optional[int] = None):
    """
    Build full graph from all processed law files.
    
    Args:
        limit: Optional limit for number of files to process (for testing)
    """
    print("=" * 60)
    print("Building Neo4j Knowledge Graph")
    print("=" * 60)
    print(f"Source: {PROCESSED_DIR}")
    print()
    
    client = get_neo4j_client()
    
    json_files = sorted(PROCESSED_DIR.glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} law files to process")
    print()
    
    total_stats = BuildStats()
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                law_data = json.load(f)
            
            stats = process_law_file(client, law_data)
            
            total_stats.laws += stats.laws
            total_stats.chapters += stats.chapters
            total_stats.articles += stats.articles
            total_stats.paragraphs += stats.paragraphs
            total_stats.errors += stats.errors
            
            if (i + 1) % 20 == 0 or (i + 1) == len(json_files):
                print(f"Progress: {i + 1}/{len(json_files)} files "
                      f"({total_stats.laws} laws, {total_stats.articles} articles)")
        
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            total_stats.errors += 1
    
    print()
    print("=" * 60)
    print("Graph Build Complete!")
    print("=" * 60)
    print(f"  Laws:       {total_stats.laws}")
    print(f"  Chapters:   {total_stats.chapters}")
    print(f"  Articles:   {total_stats.articles}")
    print(f"  Paragraphs: {total_stats.paragraphs}")
    print(f"  Errors:     {total_stats.errors}")
    print("=" * 60)
    
    # Verify in Neo4j
    print("\n📊 Verification from Neo4j:")
    node_counts = client.get_node_counts()
    for label, count in sorted(node_counts.items()):
        print(f"  {label}: {count}")
    
    client.close()
    return total_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Neo4j graph from law files")
    parser.add_argument("--limit", type=int, help="Limit number of files (for testing)")
    args = parser.parse_args()
    
    build_full_graph(limit=args.limit)
