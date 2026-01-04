#!/usr/bin/env python3
"""
Chunker for Japanese Law Documents
Processes JSON law files and creates chunks suitable for vector embedding.
Each chunk contains metadata for precise highlighting on the original text.

Output: data/chunks/
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


# Directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk, used for highlighting."""
    law_id: str
    law_title: str
    law_abbrev: Optional[str]
    category: str
    chapter_num: Optional[str]
    chapter_title: Optional[str]
    article_num: str
    article_title: str
    article_caption: Optional[str]
    paragraph_num: str
    sentence_nums: List[str]
    # Source type: "main" or "supplementary"
    source_type: str
    # For supplementary provisions
    suppl_amend_law_num: Optional[str] = None


@dataclass
class Chunk:
    """A chunk of text with metadata for embedding and retrieval."""
    chunk_id: str
    text: str
    text_with_context: str  # Includes article title/caption for better embedding
    metadata: ChunkMetadata
    char_count: int
    token_estimate: int  # Rough estimate: chars / 2 for Japanese
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "text_with_context": self.text_with_context,
            "metadata": asdict(self.metadata),
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "highlight_path": self.get_highlight_path()
        }
    
    def get_highlight_path(self) -> Dict[str, str]:
        """Get the path for highlighting in the original document."""
        path = {
            "law": self.metadata.law_title,
            "article": self.metadata.article_title
        }
        if self.metadata.chapter_title:
            path["chapter"] = self.metadata.chapter_title
        if self.metadata.paragraph_num:
            path["paragraph"] = f"{self.metadata.paragraph_num}項"
        return path


def generate_chunk_id(law_id: str, source_type: str, article_num: str, 
                      paragraph_num: str, suppl_index: Optional[int] = None) -> str:
    """Generate a unique chunk ID."""
    if source_type == "supplementary" and suppl_index is not None:
        base = f"{law_id}_suppl{suppl_index}_{article_num}_{paragraph_num}"
    else:
        base = f"{law_id}_{article_num}_{paragraph_num}"
    return base


def extract_paragraph_text(paragraph: Dict[str, Any]) -> str:
    """Extract all text from a paragraph including items."""
    texts = []
    
    # Main sentences
    for sentence in paragraph.get("sentences", []):
        if sentence.get("text"):
            texts.append(sentence["text"])
    
    # Items (一、二、三、...)
    for item in paragraph.get("items", []):
        item_title = item.get("title", "")
        for sentence in item.get("sentences", []):
            if sentence.get("text"):
                texts.append(f"{item_title} {sentence['text']}")
    
    return " ".join(texts)


def get_sentence_nums(paragraph: Dict[str, Any]) -> List[str]:
    """Get all sentence numbers in a paragraph."""
    nums = []
    for sentence in paragraph.get("sentences", []):
        if sentence.get("num"):
            nums.append(sentence["num"])
    return nums


def chunk_article(
    article: Dict[str, Any],
    law_id: str,
    law_title: str,
    law_abbrev: Optional[str],
    category: str,
    chapter_num: Optional[str],
    chapter_title: Optional[str],
    source_type: str = "main",
    suppl_amend_law_num: Optional[str] = None,
    suppl_index: Optional[int] = None
) -> List[Chunk]:
    """
    Chunk an article by paragraphs.
    Each paragraph becomes one chunk.
    """
    chunks = []
    
    article_num = article.get("num", "")
    article_title = article.get("title", f"第{article_num}条")
    article_caption = article.get("caption")
    
    for paragraph in article.get("paragraphs", []):
        paragraph_num = paragraph.get("num", "1")
        text = extract_paragraph_text(paragraph)
        
        if not text.strip():
            continue
        
        # Create context-enriched text for better embedding
        context_parts = [law_title]
        if article_caption:
            context_parts.append(article_caption)
        context_parts.append(article_title)
        if paragraph_num != "1":
            context_parts.append(f"第{paragraph_num}項")
        context_parts.append(text)
        text_with_context = " ".join(context_parts)
        
        chunk_id = generate_chunk_id(
            law_id, source_type, article_num, paragraph_num, suppl_index
        )
        
        metadata = ChunkMetadata(
            law_id=law_id,
            law_title=law_title,
            law_abbrev=law_abbrev,
            category=category,
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            article_num=article_num,
            article_title=article_title,
            article_caption=article_caption,
            paragraph_num=paragraph_num,
            sentence_nums=get_sentence_nums(paragraph),
            source_type=source_type,
            suppl_amend_law_num=suppl_amend_law_num
        )
        
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            text_with_context=text_with_context,
            metadata=metadata,
            char_count=len(text),
            token_estimate=len(text) // 2  # Rough estimate for Japanese
        )
        
        chunks.append(chunk)
    
    return chunks


def chunk_law_file(law_data: Dict[str, Any]) -> List[Chunk]:
    """
    Process a single law file and return all chunks.
    """
    chunks = []
    
    # Extract law metadata
    law_info = law_data.get("law_info", {}) or {}
    revision_info = law_data.get("revision_info", {}) or {}
    
    law_id = law_info.get("law_id", "unknown")
    law_title = revision_info.get("law_title", "")
    law_abbrev = revision_info.get("abbrev")
    category = revision_info.get("category", "")
    
    law_full_text = law_data.get("law_full_text", {}) or {}
    law_body = law_full_text.get("law_body", {}) or {}
    
    # Process main provisions
    main_provision = law_body.get("main_provision", {}) or {}
    for chapter in main_provision.get("chapters", []):
        chapter_num = chapter.get("num")
        chapter_title = chapter.get("title")
        
        for article in chapter.get("articles", []):
            article_chunks = chunk_article(
                article=article,
                law_id=law_id,
                law_title=law_title,
                law_abbrev=law_abbrev,
                category=category,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                source_type="main"
            )
            chunks.extend(article_chunks)
    
    # Process supplementary provisions
    for idx, suppl in enumerate(law_body.get("supplementary_provisions", [])):
        suppl_amend_law_num = suppl.get("amend_law_num")
        
        for article in suppl.get("articles", []):
            article_chunks = chunk_article(
                article=article,
                law_id=law_id,
                law_title=law_title,
                law_abbrev=law_abbrev,
                category=category,
                chapter_num=None,
                chapter_title="附則",
                source_type="supplementary",
                suppl_amend_law_num=suppl_amend_law_num,
                suppl_index=idx
            )
            chunks.extend(article_chunks)
    
    return chunks


def process_all_laws(save_individual: bool = True, save_combined: bool = True) -> List[Chunk]:
    """
    Process all law files and create chunks.
    
    Args:
        save_individual: Save chunks for each law as separate file
        save_combined: Save all chunks in a single file
        
    Returns:
        List of all chunks
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    stats = {
        "total_laws": 0,
        "total_chunks": 0,
        "total_chars": 0,
        "by_category": {},
        "by_law": {}
    }
    
    # Get all law JSON files (exclude index and combined files)
    json_files = [f for f in PROCESSED_DIR.glob("*.json") 
                  if not f.name.startswith("_")]
    
    print(f"Found {len(json_files)} law files to process")
    print("-" * 60)
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                law_data = json.load(f)
            
            chunks = chunk_law_file(law_data)
            
            if chunks:
                law_id = chunks[0].metadata.law_id
                law_title = chunks[0].metadata.law_title
                category = chunks[0].metadata.category
                
                # Update stats
                stats["total_laws"] += 1
                stats["total_chunks"] += len(chunks)
                stats["total_chars"] += sum(c.char_count for c in chunks)
                stats["by_category"][category] = stats["by_category"].get(category, 0) + len(chunks)
                stats["by_law"][law_id] = {
                    "title": law_title,
                    "chunk_count": len(chunks),
                    "char_count": sum(c.char_count for c in chunks)
                }
                
                print(f"✓ {law_title} ({law_id}): {len(chunks)} chunks")
                
                # Save individual chunks file
                if save_individual:
                    output_path = CHUNKS_DIR / f"{law_id}_chunks.json"
                    chunk_dicts = [c.to_dict() for c in chunks]
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
                
                all_chunks.extend(chunks)
            else:
                print(f"⚠ {json_file.name}: No chunks generated")
                
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
    
    print("-" * 60)
    print(f"\nTotal: {stats['total_chunks']} chunks from {stats['total_laws']} laws")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Average chunk size: {stats['total_chars'] // max(stats['total_chunks'], 1)} chars")
    
    print("\nChunks by category:")
    for category, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count} chunks")
    
    # Save combined chunks file
    if save_combined and all_chunks:
        combined_path = CHUNKS_DIR / "_all_chunks.json"
        all_chunk_dicts = [c.to_dict() for c in all_chunks]
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunk_dicts, f, ensure_ascii=False, indent=2)
        print(f"\nSaved combined chunks: {combined_path}")
    
    # Save stats
    stats_path = CHUNKS_DIR / "_stats.json"
    stats["processed_at"] = datetime.now().isoformat()
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved stats: {stats_path}")
    
    return all_chunks


def load_chunks(law_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load chunks from saved files.
    
    Args:
        law_id: If provided, load only chunks for this law
        
    Returns:
        List of chunk dictionaries
    """
    if law_id:
        chunk_file = CHUNKS_DIR / f"{law_id}_chunks.json"
        if chunk_file.exists():
            with open(chunk_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    else:
        combined_file = CHUNKS_DIR / "_all_chunks.json"
        if combined_file.exists():
            with open(combined_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("Japanese Law Chunker")
    print("=" * 60)
    print(f"Input directory:  {PROCESSED_DIR}")
    print(f"Output directory: {CHUNKS_DIR}")
    print("=" * 60)
    print()
    
    chunks = process_all_laws()
    
    print()
    print("=" * 60)
    print("Chunking complete!")
    print("=" * 60)
