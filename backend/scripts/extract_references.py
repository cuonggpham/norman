#!/usr/bin/env python3
"""
Extract Same-Law References from Japanese Law Documents.
Creates REFERENCES relationships between Articles that cite each other.

Run: python -m scripts.extract_references
"""

import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.neo4j_client import get_neo4j_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Reference patterns for same-law references
REFERENCE_PATTERNS = [
    # 第X条, 第X条のY (Article X, Article X-Y)
    (r'第(\d+)条(?:の(\d+))?', 'article'),
    # 前条 (Previous article)
    (r'前条', 'prev_article'),
    # 次条 (Next article)
    (r'次条', 'next_article'),
    # 第X項 within same article context
    (r'第(\d+)項', 'paragraph'),
    # 同条第X項 (Paragraph X of same article)
    (r'同条第(\d+)項', 'same_article_para'),
    # 前項 (Previous paragraph)
    (r'前項', 'prev_para'),
]


@dataclass
class Reference:
    """A reference from one article to another."""
    source_article: str
    target_article: str
    ref_type: str
    context: str


def extract_article_text(article: Dict[str, Any]) -> str:
    """Extract all text from an article."""
    texts = []
    for para in article.get("paragraphs", []):
        for sent in para.get("sentences", []):
            if sent.get("text"):
                texts.append(sent["text"])
        for item in para.get("items", []):
            for sent in item.get("sentences", []):
                if sent.get("text"):
                    texts.append(sent["text"])
    return " ".join(texts)


def parse_article_num(num_str: str) -> Optional[int]:
    """Parse article number, handling formats like '32', '32-2'."""
    if not num_str:
        return None
    match = re.search(r'(\d+)', num_str)
    return int(match.group(1)) if match else None


def find_references_in_text(text: str, current_article_num: str) -> List[Reference]:
    """
    Find article references in text.
    
    Args:
        text: Article text content
        current_article_num: Current article number (for relative refs)
    
    Returns:
        List of Reference objects
    """
    refs = []
    current_num = parse_article_num(current_article_num)
    
    for pattern, ref_type in REFERENCE_PATTERNS:
        for match in re.finditer(pattern, text):
            target_num = None
            context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
            
            if ref_type == 'article':
                # 第X条 or 第X条のY
                base_num = match.group(1)
                sub_num = match.group(2) if len(match.groups()) > 1 else None
                
                if sub_num:
                    target_num = f"{base_num}-{sub_num}"
                else:
                    target_num = base_num
                
                # Skip self-reference
                if target_num == current_article_num:
                    continue
                    
            elif ref_type == 'prev_article' and current_num:
                # 前条 → Previous article
                target_num = str(current_num - 1)
                
            elif ref_type == 'next_article' and current_num:
                # 次条 → Next article
                target_num = str(current_num + 1)
            
            if target_num and target_num != current_article_num:
                refs.append(Reference(
                    source_article=current_article_num,
                    target_article=target_num,
                    ref_type=ref_type,
                    context=context.strip()
                ))
    
    # Deduplicate
    seen = set()
    unique_refs = []
    for ref in refs:
        key = (ref.source_article, ref.target_article)
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref)
    
    return unique_refs


def create_reference_relationships(client, law_id: str, refs: List[Reference]) -> int:
    """
    Create REFERENCES relationships in Neo4j.
    
    Returns:
        Number of relationships created
    """
    query = """
    MATCH (source:Article {law_id: $law_id, num: $source_num})
    MATCH (target:Article {law_id: $law_id, num: $target_num})
    MERGE (source)-[r:REFERENCES]->(target)
    SET r.context = $context,
        r.ref_type = $ref_type
    RETURN count(r) as created
    """
    
    count = 0
    for ref in refs:
        try:
            result = client.run_query(query, {
                "law_id": law_id,
                "source_num": ref.source_article,
                "target_num": ref.target_article,
                "context": ref.context[:200] if ref.context else "",
                "ref_type": ref.ref_type
            })
            if result and result[0].get("created", 0) > 0:
                count += 1
        except Exception as e:
            # Target article may not exist, skip silently
            pass
    
    return count


def process_law_file(client, law_data: Dict[str, Any]) -> Tuple[int, int]:
    """
    Process a single law file and extract references.
    
    Returns:
        Tuple of (articles_processed, references_created)
    """
    law_id = law_data.get("law_info", {}).get("law_id", "")
    if not law_id:
        return 0, 0
    
    # Get all articles
    law_body = law_data.get("law_full_text", {}).get("law_body", {})
    main_provision = law_body.get("main_provision", {})
    
    articles = []
    # From chapters
    for chapter in main_provision.get("chapters", []):
        articles.extend(chapter.get("articles", []))
    # Direct articles
    articles.extend(main_provision.get("articles", []))
    
    articles_processed = 0
    total_refs = 0
    
    for article in articles:
        article_num = article.get("num", "")
        if not article_num:
            continue
            
        text = extract_article_text(article)
        if not text:
            continue
        
        refs = find_references_in_text(text, article_num)
        if refs:
            created = create_reference_relationships(client, law_id, refs)
            total_refs += created
        
        articles_processed += 1
    
    return articles_processed, total_refs


def extract_all_references():
    """Extract references from all law files."""
    print("=" * 60)
    print("Extracting Same-Law References")
    print("=" * 60)
    print(f"Source: {PROCESSED_DIR}")
    print()
    
    client = get_neo4j_client()
    
    json_files = [f for f in sorted(PROCESSED_DIR.glob("*.json")) 
                  if not f.name.startswith("_")]
    
    print(f"Found {len(json_files)} law files to process")
    print()
    
    total_articles = 0
    total_refs = 0
    
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                law_data = json.load(f)
            
            articles, refs = process_law_file(client, law_data)
            total_articles += articles
            total_refs += refs
            
            if (i + 1) % 50 == 0 or (i + 1) == len(json_files):
                print(f"Progress: {i + 1}/{len(json_files)} files, "
                      f"{total_refs} references found")
        
        except Exception as e:
            logger.warning(f"Error processing {json_file.name}: {e}")
    
    print()
    print("=" * 60)
    print("Reference Extraction Complete!")
    print("=" * 60)
    print(f"  Articles processed: {total_articles}")
    print(f"  REFERENCES created: {total_refs}")
    print("=" * 60)
    
    # Verify in Neo4j
    print("\n📊 Verification from Neo4j:")
    rel_counts = client.get_relationship_counts()
    for rel_type, count in sorted(rel_counts.items()):
        print(f"  {rel_type}: {count}")
    
    client.close()
    return total_refs


if __name__ == "__main__":
    extract_all_references()
