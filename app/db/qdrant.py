"""
Qdrant Vector Database Client for Cloud.

Provides functions for connecting to Qdrant Cloud,
managing collections, and performing vector operations.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()


def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant Cloud client instance.
    
    Reads QDRANT_URL and QDRANT_API_KEY from environment.
    
    Returns:
        QdrantClient connected to Qdrant Cloud
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    
    return QdrantClient(url=url, api_key=api_key)


def get_collection_name() -> str:
    """Get configured collection name."""
    return os.getenv("QDRANT_COLLECTION_NAME", "japanese_laws")


def create_collection(
    client: QdrantClient,
    collection_name: Optional[str] = None,
    vector_size: int = 3072,
    distance: Distance = Distance.COSINE,
) -> bool:
    """
    Create collection if not exists.
    
    Args:
        client: Qdrant client
        collection_name: Name of collection (default from env)
        vector_size: Dimension of vectors (3072 for text-embedding-3-large)
        distance: Distance metric (COSINE recommended)
        
    Returns:
        True if created, False if already exists
    """
    collection_name = collection_name or get_collection_name()
    
    # Check if exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        return False
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )
    return True


def upsert_vectors(
    client: QdrantClient,
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: Optional[list[int | str]] = None,
    collection_name: Optional[str] = None,
    batch_size: int = 500,
) -> int:
    """
    Upsert vectors with metadata in batches.
    
    Args:
        client: Qdrant client
        vectors: List of embedding vectors
        payloads: List of metadata dicts (same length as vectors)
        ids: Optional list of IDs (auto-generated if None)
        collection_name: Collection name (default from env)
        batch_size: Points per upsert call
        
    Returns:
        Number of points upserted
    """
    collection_name = collection_name or get_collection_name()
    
    if ids is None:
        ids = list(range(len(vectors)))
    
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        
        points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
        ]
        
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    
    return total


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 10,
    collection_name: Optional[str] = None,
    filter_conditions: Optional[dict[str, Any]] = None,
) -> list[dict]:
    """
    Perform similarity search.
    
    Args:
        client: Qdrant client
        query_vector: Query embedding
        top_k: Number of results
        collection_name: Collection name (default from env)
        filter_conditions: Optional filter (e.g., {"law_id": "322AC..."})
        
    Returns:
        List of results with id, score, and payload
    """
    collection_name = collection_name or get_collection_name()
    
    # Build filter if provided
    query_filter = None
    if filter_conditions:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_conditions.items()
        ]
        query_filter = Filter(must=conditions)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )
    
    return [
        {
            "id": r.id,
            "score": r.score,
            "payload": r.payload,
        }
        for r in results.points
    ]


def get_collection_info(
    client: QdrantClient,
    collection_name: Optional[str] = None,
) -> dict:
    """
    Get collection info including point count.
    
    Args:
        client: Qdrant client
        collection_name: Collection name (default from env)
        
    Returns:
        Dict with collection info
    """
    collection_name = collection_name or get_collection_name()
    info = client.get_collection(collection_name)
    
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "status": info.status.value,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance.value,
    }


def delete_collection(
    client: QdrantClient,
    collection_name: Optional[str] = None,
) -> bool:
    """
    Delete a collection.
    
    Args:
        client: Qdrant client
        collection_name: Collection name (default from env)
        
    Returns:
        True if deleted
    """
    collection_name = collection_name or get_collection_name()
    return client.delete_collection(collection_name)
