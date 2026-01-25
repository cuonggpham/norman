#!/usr/bin/env python3
"""
Hybrid Indexer Script (Memory-Efficient Version)

Processes embeddings FILE-BY-FILE instead of loading all at once.
This prevents memory issues on machines with limited RAM.

Usage:
    python scripts/hybrid_indexer.py [--dry-run] [--batch-size 50]
"""

import argparse
import json
import sys
import time
import gc
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant import (
    get_qdrant_client,
    get_hybrid_collection_name,
    create_hybrid_collection,
    upsert_hybrid_vectors,
    get_collection_info,
)
from app.services.sparse_embedding import SparseEmbeddingService


# Paths - use project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts -> backend -> norman
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CONFIG_FILE = EMBEDDINGS_DIR / "_config.json"
PROGRESS_FILE = EMBEDDINGS_DIR / "_hybrid_progress.json"


def get_embedding_files() -> list[tuple[Path, Path]]:
    """Get pairs of (chunks_file, embeddings_file) sorted by law_id."""
    chunks_files = sorted(EMBEDDINGS_DIR.glob("*_chunks.json"))
    pairs = []
    
    for chunks_file in chunks_files:
        law_id = chunks_file.stem.replace("_chunks", "")
        embeddings_file = EMBEDDINGS_DIR / f"{law_id}_embeddings.npy"
        
        if embeddings_file.exists():
            pairs.append((chunks_file, embeddings_file))
    
    return pairs


def load_progress() -> dict:
    """Load indexing progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "completed_files": [],
        "total_indexed": 0,
        "current_global_id": 0,
    }


def save_progress(progress: dict):
    """Save indexing progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def prepare_payloads(chunks: list[dict]) -> list[dict]:
    """Convert chunk data to Qdrant payloads."""
    payloads = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        payload = {
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
        payloads.append(payload)
    return payloads


def process_single_file(
    chunks_file: Path,
    embeddings_file: Path,
    client,
    collection_name: str,
    sparse_service: SparseEmbeddingService,
    start_id: int,
    batch_size: int = 20,
    sparse_batch_size: int = 16,
    delay_between_batches: float = 0.5,
) -> int:
    """
    Process a single law file (ultra memory-efficient).
    
    Uses memory-mapped numpy arrays and processes in very small batches
    to prevent machine freezing.
    
    Returns:
        Number of vectors indexed
    """
    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Use memory-mapped mode to avoid loading full array into RAM
    embeddings = np.load(embeddings_file, mmap_mode='r')
    
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Mismatch in {chunks_file.name}: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")
    
    total_count = len(chunks)
    
    # Process in small batches to minimize memory usage
    total_indexed = 0
    max_retries = 3
    
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        
        # Extract only this batch's data (not all at once)
        batch_chunks = chunks[i:end_idx]
        batch_payloads = prepare_payloads(batch_chunks)
        batch_texts = [c.get("text_with_context", c.get("text", "")) for c in batch_chunks]
        
        # Convert only this batch's embeddings to list (mmap reads only needed portion)
        batch_dense = embeddings[i:end_idx].tolist()
        batch_ids = list(range(start_id + i, start_id + end_idx))
        
        # Generate sparse embeddings for this batch only
        batch_sparse = sparse_service.embed_batch(batch_texts, batch_size=sparse_batch_size)
        
        # Upload with retry logic
        for attempt in range(max_retries):
            try:
                upsert_hybrid_vectors(
                    client=client,
                    dense_vectors=batch_dense,
                    sparse_vectors=batch_sparse,
                    payloads=batch_payloads,
                    ids=batch_ids,
                    collection_name=collection_name,
                    batch_size=len(batch_dense),
                )
                total_indexed += len(batch_ids)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    tqdm.write(f"  Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        
        # Cleanup batch data and add delay to prevent system overload
        del batch_chunks, batch_payloads, batch_texts, batch_dense, batch_sparse
        gc.collect()
        time.sleep(delay_between_batches)
    
    # Cleanup
    del chunks, embeddings
    gc.collect()
    
    return total_indexed


def run_hybrid_indexer(
    batch_size: int = 20,
    dry_run: bool = False,
    resume: bool = True,
    sparse_batch_size: int = 16,
    delay_between_batches: float = 0.5,
):
    """
    Memory-efficient hybrid indexing pipeline.
    
    Processes files one-by-one instead of loading all data.
    """
    # Get all file pairs
    file_pairs = get_embedding_files()
    print(f"Found {len(file_pairs)} law files to process")
    
    # Count total chunks
    if dry_run:
        total_chunks = 0
        for chunks_file, _ in file_pairs:
            with open(chunks_file, "r") as f:
                total_chunks += len(json.load(f))
        
        print(f"\n[DRY RUN] Would upload:")
        print(f"  Total files: {len(file_pairs)}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Batch size: {batch_size}")
        return
    
    # Load config
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    print(f"Config: model={config['model']}, dims={config['dimensions']}")
    
    # Load progress
    progress = load_progress() if resume else {
        "completed_files": [],
        "total_indexed": 0,
        "current_global_id": 0,
    }
    
    completed_files = set(progress.get("completed_files", []))
    current_id = progress.get("current_global_id", 0)
    
    if completed_files:
        print(f"Resuming: {len(completed_files)} files already completed")
    
    # Filter out completed files
    remaining_pairs = [
        (c, e) for c, e in file_pairs 
        if c.stem.replace("_chunks", "") not in completed_files
    ]
    
    if not remaining_pairs:
        print("All files already indexed!")
        return
    
    print(f"Files remaining: {len(remaining_pairs)}")
    
    # Initialize services
    print("\nInitializing sparse embedding service...")
    sparse_service = SparseEmbeddingService()
    print(f"  Model: {sparse_service.model_name}")
    
    print("\nConnecting to Qdrant Cloud...")
    client = get_qdrant_client()
    collection_name = get_hybrid_collection_name()
    
    # Create collection
    print(f"Creating hybrid collection '{collection_name}'...")
    created = create_hybrid_collection(
        client=client,
        collection_name=collection_name,
        dense_size=config["dimensions"],
    )
    print(f"  {'Created' if created else 'Already exists'}")
    
    # Process files one by one
    print(f"\nProcessing {len(remaining_pairs)} files...")
    
    for chunks_file, embeddings_file in tqdm(remaining_pairs, desc="Files"):
        law_id = chunks_file.stem.replace("_chunks", "")
        
        try:
            indexed = process_single_file(
                chunks_file=chunks_file,
                embeddings_file=embeddings_file,
                client=client,
                collection_name=collection_name,
                sparse_service=sparse_service,
                start_id=current_id,
                batch_size=batch_size,
                sparse_batch_size=sparse_batch_size,
                delay_between_batches=delay_between_batches,
            )
            
            # Update progress
            current_id += indexed
            progress["completed_files"].append(law_id)
            progress["total_indexed"] = current_id
            progress["current_global_id"] = current_id
            save_progress(progress)
            
        except Exception as e:
            tqdm.write(f"Error processing {law_id}: {e}")
            raise
    
    print(f"\n✓ Indexed {current_id} vectors to hybrid collection")
    
    # Verify
    print("\nVerifying...")
    info = get_collection_info(client, collection_name)
    print(f"  Collection: {info['name']}")
    print(f"  Points: {info['points_count']}")
    print(f"  Status: {info['status']}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient hybrid indexing (file-by-file)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files and chunks without uploading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Vectors per batch (default: 20, smaller = less RAM)",
    )
    parser.add_argument(
        "--sparse-batch-size",
        type=int,
        default=16,
        help="Batch size for sparse embedding (default: 16)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from progress",
    )
    args = parser.parse_args()
    
    run_hybrid_indexer(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        sparse_batch_size=args.sparse_batch_size,
        delay_between_batches=args.delay,
    )


if __name__ == "__main__":
    main()
