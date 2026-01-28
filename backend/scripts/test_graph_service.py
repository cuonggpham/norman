#!/usr/bin/env python3
"""
Test Graph Service and Query Router.
Verifies the GraphRAG components work correctly.

Run: python -m scripts.test_graph_service
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.graph_service import get_graph_service
from app.services.query_router import get_query_router, QueryType


def test_graph_stats():
    """Test getting graph statistics."""
    print("\n" + "=" * 60)
    print("1. Graph Statistics")
    print("=" * 60)
    
    service = get_graph_service()
    stats = service.get_graph_stats()
    
    print("Nodes:")
    for label, count in sorted(stats.get("nodes", {}).items()):
        print(f"  {label}: {count}")
    
    print("\nRelationships:")
    for rel_type, count in sorted(stats.get("relationships", {}).items()):
        print(f"  {rel_type}: {count}")


def test_find_article():
    """Test finding specific article."""
    print("\n" + "=" * 60)
    print("2. Find Article: 労働基準法 第32条")
    print("=" * 60)
    
    service = get_graph_service()
    result = service.find_article("労働基準法", "32")
    
    if result:
        print(f"✅ Found!")
        print(f"  Law ID: {result.law_id}")
        print(f"  Law: {result.law_title}")
        print(f"  Article: 第{result.article_num}条")
        print(f"  Title: {result.article_title}")
        print(f"  Caption: {result.article_caption}")
        print(f"  Chunk ID: {result.chunk_id}")
        print(f"  Path: {' → '.join(result.path)}")
    else:
        print("❌ Not found")


def test_related_articles():
    """Test finding related articles via REFERENCES."""
    print("\n" + "=" * 60)
    print("3. Find Related Articles (via REFERENCES)")
    print("=" * 60)
    
    service = get_graph_service()
    
    # First, find a law_id that has references
    test_query = """
    MATCH (a:Article)-[r:REFERENCES]->(b:Article)
    RETURN a.law_id as law_id, a.num as article_num, count(r) as ref_count
    ORDER BY ref_count DESC
    LIMIT 1
    """
    
    from app.db.neo4j import get_neo4j_client
    client = get_neo4j_client()
    results = client.run_query(test_query)
    
    if results:
        law_id = results[0]["law_id"]
        article_num = results[0]["article_num"]
        ref_count = results[0]["ref_count"]
        
        print(f"Testing with: {law_id} 第{article_num}条 ({ref_count} outgoing refs)")
        
        related = service.find_related_articles(law_id, article_num, depth=2, limit=5)
        
        if related:
            print(f"\n✅ Found {len(related)} related articles:")
            for i, r in enumerate(related, 1):
                print(f"  {i}. 第{r.article_num}条 - {r.article_title or r.article_caption or 'N/A'}")
                print(f"     Relevance: {r.relevance:.2f}")
        else:
            print("❌ No related articles found")
    else:
        print("⚠️ No articles with references found")


def test_keyword_search():
    """Test keyword search."""
    print("\n" + "=" * 60)
    print("4. Keyword Search: '労働時間'")
    print("=" * 60)
    
    service = get_graph_service()
    results = service.search_by_keyword("労働時間", limit=5)
    
    if results:
        print(f"✅ Found {len(results)} articles:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.law_title} 第{r.article_num}条")
            print(f"     Title: {r.article_title or r.article_caption or 'N/A'}")
    else:
        print("❌ No results found")


def test_query_router():
    """Test query routing."""
    print("\n" + "=" * 60)
    print("5. Query Router Tests")
    print("=" * 60)
    
    router = get_query_router()
    
    test_queries = [
        "労働基準法第32条について教えてください",
        "第37条 nói gì?",
        "Các điều liên quan đến thời gian làm việc",
        "Thuế thu nhập cho người nước ngoài tính như thế nào?",
        "所得税法の規定",
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: {query[:40]}...")
        print(f"  Type: {result.query_type.value}")
        print(f"  Entities: {result.entities}")
        print(f"  Use Graph: {result.use_graph}")
        print(f"  Use Vector: {result.use_vector}")


def test_law_structure():
    """Test getting law structure."""
    print("\n" + "=" * 60)
    print("6. Law Structure: 労働基準法")
    print("=" * 60)
    
    service = get_graph_service()
    
    # First find a law
    from app.db.neo4j import get_neo4j_client
    client = get_neo4j_client()
    results = client.run_query(
        "MATCH (l:Law) WHERE l.title CONTAINS '労働基準' RETURN l.law_id LIMIT 1"
    )
    
    if results:
        law_id = results[0]["l.law_id"]
        structure = service.get_law_structure(law_id)
        
        if structure:
            print(f"✅ Law: {structure.get('law_title')}")
            print(f"   Chapters: {len(structure.get('chapters', []))}")
            
            for ch in structure.get('chapters', [])[:3]:  # Show first 3
                print(f"   - 第{ch['num']}章: {ch['title']}")
                print(f"     Articles: {len(ch.get('articles', []))}")
        else:
            print("❌ Structure not found")
    else:
        print("⚠️ 労働基準法 not found")


def main():
    print("=" * 60)
    print("GraphRAG Service Tests")
    print("=" * 60)
    
    try:
        test_graph_stats()
        test_find_article()
        test_related_articles()
        test_keyword_search()
        test_query_router()
        test_law_structure()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
