#!/usr/bin/env python3
"""
Embedder for Japanese Law Chunks
Creates embeddings using OpenAI's text-embedding-3-large model.
Supports batch processing, resume capability, and progress tracking.

Output: data/embeddings/
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Directories - use project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts -> backend -> norman
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def load_chunks(law_id: Optional[str] = None) -> List[Dict]:
    """
    Load chunks from data/chunks/.
    
    Args:
        law_id: If provided, load only chunks for this law
        
    Returns:
        List of chunk dictionaries
    """
    if law_id:
        chunk_file = CHUNKS_DIR / f"{law_id}_chunks.json"
        if not chunk_file.exists():
            print(f"Warning: {chunk_file} not found")
            return []
        with open(chunk_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Load all chunks from _all_chunks.json
        all_chunks_file = CHUNKS_DIR / "_all_chunks.json"
        if all_chunks_file.exists():
            with open(all_chunks_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Fallback: load from individual files
        all_chunks = []
        for chunk_file in sorted(CHUNKS_DIR.glob("*_chunks.json")):
            if chunk_file.name.startswith("_"):
                continue
            with open(chunk_file, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
        return all_chunks


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens for Japanese/mixed text."""
    # Japanese characters are typically 1-2 tokens each
    # This is a conservative estimate
    return len(text) // 2


def truncate_text(text: str, max_tokens: int = 6000) -> str:
    """Truncate text to fit within token limit."""
    # Japanese characters can be 2-3 tokens each
    # Using conservative ratio: 1 char = 3 tokens
    max_chars = max_tokens // 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars]



def embed_single(text: str) -> np.ndarray:
    """Embed a single text, with truncation if needed."""
    truncated = truncate_text(text)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[truncated],
        dimensions=EMBEDDING_DIMENSIONS
    )
    return np.array([response.data[0].embedding], dtype=np.float32)


def embed_batch(texts: List[str], retry_count: int = 0) -> np.ndarray:
    """
    Embed a batch of texts using OpenAI API.
    Handles token overflow by splitting batch or processing individually.
    
    Args:
        texts: List of texts to embed
        retry_count: Current retry attempt
        
    Returns:
        numpy array of shape (batch_size, dimensions)
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a token overflow error (400 error with context length message)
        if "maximum context length" in error_str or "8192 tokens" in error_str:
            if len(texts) > 1:
                # Split batch in half and try again
                mid = len(texts) // 2
                tqdm.write(f"  Token overflow with {len(texts)} texts, splitting...")
                first_half = embed_batch(texts[:mid])
                second_half = embed_batch(texts[mid:])
                return np.vstack([first_half, second_half])
            else:
                # Single text too long, truncate it
                tqdm.write(f"  Truncating long text ({len(texts[0])} chars)...")
                return embed_single(texts[0])
        
        # For other errors, retry with backoff
        if retry_count < MAX_RETRIES:
            print(f"\nError: {e}")
            print(f"Retrying in {RETRY_DELAY}s... (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
            return embed_batch(texts, retry_count + 1)
        else:
            raise Exception(f"Failed after {MAX_RETRIES} retries: {e}")


def get_completed_laws() -> set:
    """Get set of law IDs that have already been processed."""
    progress_file = EMBEDDINGS_DIR / "_progress.json"
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)
            return set(progress.get("completed_laws", []))
    return set()


def save_progress(completed_laws: set, total_chunks: int, total_tokens: int):
    """Save progress checkpoint."""
    progress_file = EMBEDDINGS_DIR / "_progress.json"
    progress = {
        "completed_laws": list(completed_laws),
        "total_chunks_processed": total_chunks,
        "total_tokens_estimated": total_tokens,
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def process_law(law_id: str, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Process a single law file.
    
    Args:
        law_id: Law identifier
        chunks: List of chunks for this law
        
    Returns:
        Tuple of (embeddings array, chunks with indices)
    """
    # Extract texts for embedding (use text_with_context)
    texts = [chunk["text_with_context"] for chunk in chunks]
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = embed_batch(batch_texts)
        all_embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # Add embedding index to chunks
    for idx, chunk in enumerate(chunks):
        chunk["embedding_index"] = idx
    
    return embeddings, chunks


def save_law_embeddings(law_id: str, embeddings: np.ndarray, chunks: List[Dict]):
    """Save embeddings and chunks for a single law."""
    # Save embeddings as numpy array
    np.save(EMBEDDINGS_DIR / f"{law_id}_embeddings.npy", embeddings)
    
    # Save chunks with indices
    with open(EMBEDDINGS_DIR / f"{law_id}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def process_all(resume: bool = True) -> None:
    """
    Main entry point - process all laws.
    
    Args:
        resume: If True, skip already processed laws
    """
    # Ensure output directory exists
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of law files
    law_files = sorted([f for f in CHUNKS_DIR.glob("*_chunks.json") 
                        if not f.name.startswith("_")])
    
    if not law_files:
        print("No chunk files found!")
        return
    
    # Check for resume
    completed_laws = get_completed_laws() if resume else set()
    if completed_laws:
        print(f"Resuming... {len(completed_laws)} laws already processed")
    
    # Track progress
    all_embeddings = []
    all_chunks = []
    total_tokens = 0
    
    # Load existing embeddings if resuming
    if resume and completed_laws:
        for law_id in sorted(completed_laws):
            emb_file = EMBEDDINGS_DIR / f"{law_id}_embeddings.npy"
            chunk_file = EMBEDDINGS_DIR / f"{law_id}_chunks.json"
            if emb_file.exists() and chunk_file.exists():
                all_embeddings.append(np.load(emb_file))
                with open(chunk_file, "r", encoding="utf-8") as f:
                    all_chunks.extend(json.load(f))
    
    # Process each law
    for law_file in tqdm(law_files, desc="Processing laws"):
        law_id = law_file.stem.replace("_chunks", "")
        
        # Skip if already processed
        if law_id in completed_laws:
            continue
        
        # Load chunks
        with open(law_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            continue
        
        tqdm.write(f"Processing {law_id}: {len(chunks)} chunks")
        
        # Process law
        embeddings, chunks_with_idx = process_law(law_id, chunks)
        
        # Save individual law files
        save_law_embeddings(law_id, embeddings, chunks_with_idx)
        
        # Accumulate
        all_embeddings.append(embeddings)
        all_chunks.extend(chunks_with_idx)
        
        # Estimate tokens (rough: chars / 4 for Japanese)
        law_tokens = sum(len(c["text_with_context"]) for c in chunks) // 4
        total_tokens += law_tokens
        
        # Update progress
        completed_laws.add(law_id)
        save_progress(completed_laws, len(all_chunks), total_tokens)
    
    # Save combined files
    if all_embeddings:
        print("\nSaving combined embeddings...")
        combined_embeddings = np.vstack(all_embeddings)
        np.save(EMBEDDINGS_DIR / "_all_embeddings.npy", combined_embeddings)
        
        # Update indices for combined file
        for idx, chunk in enumerate(all_chunks):
            chunk["embedding_index"] = idx
        
        with open(EMBEDDINGS_DIR / "_all_chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Save config
        config = {
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "batch_size": BATCH_SIZE,
            "total_chunks": len(all_chunks),
            "total_tokens_estimated": total_tokens,
            "created_at": datetime.now().isoformat()
        }
        with open(EMBEDDINGS_DIR / "_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Total laws:     {len(completed_laws)}")
    print(f"Total chunks:   {len(all_chunks)}")
    print(f"Dimensions:     {EMBEDDING_DIMENSIONS}")
    print(f"Est. tokens:    {total_tokens:,}")
    print(f"Model:          {EMBEDDING_MODEL}")
    if all_embeddings:
        size_mb = combined_embeddings.nbytes / 1024 / 1024
        print(f"Embeddings:     {combined_embeddings.shape} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Japanese Law Embedder")
    print("=" * 60)
    print(f"Model:       {EMBEDDING_MODEL}")
    print(f"Dimensions:  {EMBEDDING_DIMENSIONS}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Input:       {CHUNKS_DIR}")
    print(f"Output:      {EMBEDDINGS_DIR}")
    print("=" * 60)
    print()
    
    process_all(resume=True)
