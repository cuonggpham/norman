#!/usr/bin/env python3
"""
Indexer Script - Upload embeddings to Qdrant Cloud.

Loads pre-generated embeddings and uploads them to Qdrant Cloud
with metadata for semantic search.

Usage:
    python scripts/indexer.py [--dry-run] [--batch-size 500]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant import (
    get_qdrant_client,
    get_collection_name,
    create_collection,
    upsert_vectors,
    get_collection_info,
)


# Paths - use project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts -> backend -> norman
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CHUNKS_FILE = EMBEDDINGS_DIR / "_all_chunks.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "_all_embeddings.npy"
CONFIG_FILE = EMBEDDINGS_DIR / "_config.json"


def load_data() -> tuple[list[dict], np.ndarray, dict]:
    """Load chunks, embeddings, and config."""
    print("Loading data...")
    
    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Chunks: {len(chunks)}")
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  Embeddings: {embeddings.shape}")
    
    # Load config
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"  Model: {config['model']}, Dims: {config['dimensions']}")
    
    # Verify alignment
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )
    
    return chunks, embeddings, config


def prepare_payloads(chunks: list[dict]) -> list[dict]:
    """
    Convert chunk data to Qdrant payloads.
    
    Keeps essential metadata for filtering and display.
    """
    payloads = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        payload = {
            # IDs
            "chunk_id": chunk.get("chunk_id", ""),
            "law_id": meta.get("law_id", ""),
            # Law info
            "law_title": meta.get("law_title", ""),
            "law_abbrev": meta.get("law_abbrev", ""),
            "law_type": meta.get("law_type", ""),  # Act, CabinetOrder, etc.
            "category": meta.get("category", ""),
            # Structure
            "chapter_num": meta.get("chapter_num", ""),
            "chapter_title": meta.get("chapter_title", ""),
            "article_num": meta.get("article_num", ""),
            "article_title": meta.get("article_title", ""),
            "article_caption": meta.get("article_caption", ""),
            "paragraph_num": meta.get("paragraph_num", ""),
            "source_type": meta.get("source_type", "main"),
            # Text content
            "text": chunk.get("text", ""),
            "text_with_context": chunk.get("text_with_context", ""),
            # Stats
            "char_count": chunk.get("char_count", 0),
        }
        payloads.append(payload)
    return payloads


def run_indexer(batch_size: int = 500, dry_run: bool = False):
    """
    Main indexing pipeline.
    
    Args:
        batch_size: Number of vectors per upsert call
        dry_run: If True, load data and show stats but don't upload
    """
    # Load data
    chunks, embeddings, config = load_data()
    
    if dry_run:
        print("\n[DRY RUN] Would upload:")
        print(f"  Vectors: {len(chunks)}")
        print(f"  Dimensions: {config['dimensions']}")
        print(f"  Batch size: {batch_size}")
        print(f"  Estimated batches: {(len(chunks) + batch_size - 1) // batch_size}")
        return
    
    # Connect to Qdrant Cloud
    print("\nConnecting to Qdrant Cloud...")
    client = get_qdrant_client()
    collection_name = get_collection_name()
    
    # Create collection
    print(f"Creating collection '{collection_name}'...")
    created = create_collection(
        client=client,
        collection_name=collection_name,
        vector_size=config["dimensions"],
    )
    if created:
        print("  Collection created.")
    else:
        print("  Collection already exists.")
    
    # Prepare payloads
    print("Preparing payloads...")
    payloads = prepare_payloads(chunks)
    
    # Convert embeddings to list
    vectors = embeddings.tolist()
    
    # Generate IDs (using index)
    ids = list(range(len(vectors)))
    
    # Upload in batches with progress bar and retry
    print(f"\nUploading {len(vectors)} vectors in batches of {batch_size}...")
    total_uploaded = 0
    max_retries = 3
    
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading"):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                count = upsert_vectors(
                    client=client,
                    vectors=batch_vectors,
                    payloads=batch_payloads,
                    ids=batch_ids,
                    collection_name=collection_name,
                    batch_size=len(batch_vectors),
                )
                total_uploaded += count
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 5
                    tqdm.write(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"  Failed batch at index {i}: {e}")
                    raise
    
    print(f"\nUploaded {total_uploaded} vectors.")
    
    # Verify
    print("\nVerifying...")
    info = get_collection_info(client, collection_name)
    print(f"  Collection: {info['name']}")
    print(f"  Points: {info['points_count']}")
    print(f"  Status: {info['status']}")
    
    # Check count matches
    expected = len(chunks)
    actual = info["points_count"]
    if actual == expected:
        print(f"\n✓ Success! All {expected} vectors indexed.")
    else:
        print(f"\n⚠ Warning: Expected {expected}, got {actual} vectors.")


def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant Cloud")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data without uploading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Vectors per batch (default: 50)",
    )
    args = parser.parse_args()
    
    run_indexer(batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
