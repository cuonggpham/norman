#!/usr/bin/env python3
"""
Streaming Indexer - Memory-efficient upload to Qdrant Cloud.

Instead of loading all 1.1GB embeddings at once, this script
processes each law file individually to minimize memory usage.

Usage:
    python scripts/indexer_streaming.py [--dry-run] [--batch-size 50]
"""

import argparse
import json
import os
import sys
import time
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


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CONFIG_FILE = EMBEDDINGS_DIR / "_config.json"
PROGRESS_FILE = EMBEDDINGS_DIR / "_index_progress.json"


def get_law_files() -> list[tuple[Path, Path]]:
    """Get list of (chunks_file, embeddings_file) tuples for each law."""
    files = []
    for chunk_file in sorted(EMBEDDINGS_DIR.glob("*_chunks.json")):
        if chunk_file.name.startswith("_"):
            continue
        law_id = chunk_file.stem.replace("_chunks", "")
        emb_file = EMBEDDINGS_DIR / f"{law_id}_embeddings.npy"
        if emb_file.exists():
            files.append((chunk_file, emb_file))
    return files


def prepare_payload(chunk: dict) -> dict:
    """Convert a single chunk to Qdrant payload."""
    meta = chunk.get("metadata", {})
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "law_id": meta.get("law_id", ""),
        "law_title": meta.get("law_title", ""),
        "law_abbrev": meta.get("law_abbrev", ""),
        "law_type": meta.get("law_type", ""),
        "category": meta.get("category", ""),
        "chapter_num": meta.get("chapter_num", ""),
        "chapter_title": meta.get("chapter_title", ""),
        "article_num": meta.get("article_num", ""),
        "article_title": meta.get("article_title", ""),
        "article_caption": meta.get("article_caption", ""),
        "paragraph_num": meta.get("paragraph_num", ""),
        "source_type": meta.get("source_type", "main"),
        "text": chunk.get("text", ""),
        "text_with_context": chunk.get("text_with_context", ""),
        "char_count": chunk.get("char_count", 0),
    }


def load_progress() -> dict:
    """Load indexing progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_laws": [], "total_vectors": 0, "next_id": 0}


def save_progress(progress: dict):
    """Save indexing progress."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def process_law_file(
    client,
    collection_name: str,
    chunk_file: Path,
    emb_file: Path,
    start_id: int,
    batch_size: int = 50,
    max_retries: int = 3,
) -> int:
    """
    Process a single law file and upload to Qdrant.
    
    Returns: Number of vectors uploaded
    """
    # Load chunks and embeddings for this law only
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    embeddings = np.load(emb_file)
    
    if len(chunks) != embeddings.shape[0]:
        print(f"  ⚠️ Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")
        return 0
    
    # Prepare payloads
    payloads = [prepare_payload(c) for c in chunks]
    vectors = embeddings.tolist()
    ids = list(range(start_id, start_id + len(vectors)))
    
    # Upload in batches
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        
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
                total += count
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
    
    return total


def run_indexer(batch_size: int = 50, dry_run: bool = False, resume: bool = True):
    """Main indexing pipeline with streaming."""
    
    # Load config
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    
    print(f"Model: {config['model']}")
    print(f"Dimensions: {config['dimensions']}")
    
    # Get law files
    law_files = get_law_files()
    print(f"Found {len(law_files)} law files to process")
    
    if dry_run:
        total_chunks = 0
        for chunk_file, emb_file in law_files:
            with open(chunk_file, "r") as f:
                chunks = json.load(f)
            total_chunks += len(chunks)
        print(f"\n[DRY RUN] Would upload {total_chunks} vectors")
        print(f"Estimated batches: {(total_chunks + batch_size - 1) // batch_size}")
        return
    
    # Load progress
    progress = load_progress() if resume else {"completed_laws": [], "total_vectors": 0, "next_id": 0}
    completed = set(progress["completed_laws"])
    
    if completed:
        print(f"Resuming... {len(completed)} laws already indexed")
    
    # Connect to Qdrant
    print("\nConnecting to Qdrant Cloud...")
    client = get_qdrant_client()
    collection_name = get_collection_name()
    
    # Create collection if needed
    created = create_collection(
        client=client,
        collection_name=collection_name,
        vector_size=config["dimensions"],
    )
    if created:
        print(f"Created collection '{collection_name}'")
    else:
        print(f"Collection '{collection_name}' already exists")
    
    # Process each law file
    total_uploaded = progress["total_vectors"]
    next_id = progress["next_id"]
    
    for chunk_file, emb_file in tqdm(law_files, desc="Indexing"):
        law_id = chunk_file.stem.replace("_chunks", "")
        
        if law_id in completed:
            continue
        
        try:
            # Load and get count first
            with open(chunk_file, "r") as f:
                chunks = json.load(f)
            num_chunks = len(chunks)
            del chunks  # Free memory
            
            tqdm.write(f"  {law_id}: {num_chunks} vectors")
            
            count = process_law_file(
                client=client,
                collection_name=collection_name,
                chunk_file=chunk_file,
                emb_file=emb_file,
                start_id=next_id,
                batch_size=batch_size,
            )
            
            total_uploaded += count
            next_id += count
            completed.add(law_id)
            
            # Save progress after each law
            progress = {
                "completed_laws": list(completed),
                "total_vectors": total_uploaded,
                "next_id": next_id,
            }
            save_progress(progress)
            
        except Exception as e:
            tqdm.write(f"  ❌ Error processing {law_id}: {e}")
            # Save progress and exit
            save_progress(progress)
            raise
    
    # Verify
    print(f"\nUploaded {total_uploaded} vectors")
    print("\nVerifying...")
    info = get_collection_info(client, collection_name)
    print(f"  Collection: {info['name']}")
    print(f"  Points: {info['points_count']}")
    print(f"  Status: {info['status']}")
    
    if info["points_count"] >= total_uploaded:
        print(f"\n✅ Success! {info['points_count']} vectors indexed.")
    else:
        print(f"\n⚠️ Warning: Expected >= {total_uploaded}, got {info['points_count']}")


def main():
    parser = argparse.ArgumentParser(description="Streaming upload to Qdrant")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count vectors without uploading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Vectors per upload batch (default: 50)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous progress",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Streaming Indexer - Memory Efficient")
    print("=" * 60)
    
    run_indexer(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
