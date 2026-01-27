#!/usr/bin/env python3
"""
Initialize Graph Schema in Neo4j.
Creates constraints and indexes for optimal query performance.

Run: python -m scripts.init_graph_schema
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.neo4j_client import get_neo4j_client


# Schema definition queries
SCHEMA_QUERIES = [
    # ============================================
    # Unique Constraints
    # ============================================
    # Law nodes must have unique law_id
    "CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.law_id IS UNIQUE",
    
    # Paragraph nodes must have unique chunk_id (links to vector store)
    "CREATE CONSTRAINT paragraph_chunk_id IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.chunk_id IS UNIQUE",
    
    # LegalTerm nodes must have unique term name
    "CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:LegalTerm) REQUIRE t.term IS UNIQUE",
    
    # ============================================
    # Indexes for Fast Lookup
    # ============================================
    # Index on Law title for text search
    "CREATE INDEX law_title IF NOT EXISTS FOR (l:Law) ON (l.title)",
    
    # Index on Article num for quick lookup
    "CREATE INDEX article_num IF NOT EXISTS FOR (a:Article) ON (a.num)",
    
    # Composite index for Article lookup by law_id + num
    "CREATE INDEX article_law_num IF NOT EXISTS FOR (a:Article) ON (a.law_id, a.num)",
    
    # Index on Chapter for navigation
    "CREATE INDEX chapter_law IF NOT EXISTS FOR (c:Chapter) ON (c.law_id)",
]


def init_schema():
    """Initialize Neo4j schema with constraints and indexes."""
    print("=" * 60)
    print("Initializing Neo4j Graph Schema")
    print("=" * 60)
    
    client = get_neo4j_client()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for query in SCHEMA_QUERIES:
        try:
            client.run_write(query)
            # Extract constraint/index name for display
            name = query.split("IF NOT EXISTS")[0].split("CONSTRAINT")[-1].split("INDEX")[-1].strip()
            print(f"✅ Created: {name}")
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                print(f"⏭️  Already exists: {query[:40]}...")
                skip_count += 1
            else:
                print(f"❌ Error: {error_msg}")
                error_count += 1
    
    print()
    print("=" * 60)
    print(f"Schema initialization complete!")
    print(f"  Created: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors:  {error_count}")
    print("=" * 60)
    
    client.close()
    return error_count == 0


if __name__ == "__main__":
    success = init_schema()
    sys.exit(0 if success else 1)
