#!/usr/bin/env python3
"""
Test Neo4j Aura Connection.
Run: python -m scripts.test_neo4j
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.neo4j_client import get_neo4j_client
from app.core.config import get_settings


def main():
    settings = get_settings()
    
    print("=" * 60)
    print("Neo4j Aura Connection Test")
    print("=" * 60)
    print(f"URI: {settings.neo4j_uri}")
    print(f"User: {settings.neo4j_user}")
    print(f"Database: {settings.neo4j_database}")
    print("=" * 60)
    print()
    
    # Get client
    client = get_neo4j_client()
    
    # Test connection
    print("Testing connection...")
    if client.verify_connection():
        print("✅ Neo4j connection successful!")
        
        # Test query
        result = client.run_query("RETURN 'Hello GraphRAG!' AS message")
        print(f"   Message: {result[0]['message']}")
        
        # Get current stats
        print("\n📊 Current Graph Statistics:")
        node_counts = client.get_node_counts()
        if node_counts:
            for label, count in node_counts.items():
                print(f"   {label}: {count}")
        else:
            print("   (Graph is empty)")
        
        rel_counts = client.get_relationship_counts()
        if rel_counts:
            print("\n   Relationships:")
            for rel_type, count in rel_counts.items():
                print(f"   {rel_type}: {count}")
        
    else:
        print("❌ Neo4j connection FAILED!")
        print("   Please check your credentials in .env")
        return 1
    
    print()
    print("=" * 60)
    print("Connection test complete!")
    print("=" * 60)
    
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
