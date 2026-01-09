#!/usr/bin/env python3
"""
Create Qdrant payload indexes for filtering.

Usage:
    python scripts/create_indexes.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.models import PayloadSchemaType

from app.db.qdrant import get_qdrant_client, get_collection_name


def create_category_index(client, collection_name: str) -> bool:
    """
    Create keyword index for 'category' field.
    
    This enables filtering by category in search queries.
    
    Args:
        client: Qdrant client
        collection_name: Name of collection
        
    Returns:
        True if created successfully
    """
    print(f"Creating index for 'category' field in '{collection_name}'...")
    
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("✅ 'category' index created successfully!")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️ 'category' index already exists")
            return True
        print(f"❌ Error creating index: {e}")
        return False


def create_law_title_index(client, collection_name: str) -> bool:
    """
    Create keyword index for 'law_title' field.
    
    This enables filtering by law name in search queries.
    """
    print(f"Creating index for 'law_title' field in '{collection_name}'...")
    
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="law_title",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("✅ 'law_title' index created successfully!")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️ 'law_title' index already exists")
            return True
        print(f"❌ Error creating index: {e}")
        return False


def main():
    """Create all payload indexes."""
    print("=" * 60)
    print("Qdrant Payload Index Creator")
    print("=" * 60)
    
    # Connect to Qdrant
    print("\n🔌 Connecting to Qdrant...")
    client = get_qdrant_client()
    collection_name = get_collection_name()
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"✅ Connected to '{collection_name}'")
    print(f"   Points: {collection_info.points_count}")
    
    # Create indexes
    print("\n📝 Creating indexes...")
    results = []
    
    results.append(("category", create_category_index(client, collection_name)))
    results.append(("law_title", create_law_title_index(client, collection_name)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for field, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {field}")
    
    print("\n" + "=" * 60)
    
    if all(success for _, success in results):
        print("All indexes created successfully!")
        print("\nNext step: Enable auto_filter in rag.py")
        return 0
    else:
        print("Some indexes failed to create")
        return 1


if __name__ == "__main__":
    sys.exit(main())
