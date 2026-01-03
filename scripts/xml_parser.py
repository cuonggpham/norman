#!/usr/bin/env python3
"""
XML Parser for Japanese Law Documents
Parses raw XML files from data/xml_raw and converts them to structured JSON
Output is saved to data/processed
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


# Directories
BASE_DIR = Path(__file__).parent
RAW_XML_DIR = BASE_DIR / "data" / "xml_raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def ensure_processed_dir():
    """Create processed directory if it doesn't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_element_text(element: Optional[ET.Element]) -> Optional[str]:
    """Extract text from an element, handling None cases."""
    if element is None:
        return None
    return element.text.strip() if element.text else None


def parse_attributes(element: ET.Element) -> Dict[str, str]:
    """Extract all attributes from an element."""
    return dict(element.attrib)


def parse_sentence(sentence: ET.Element) -> Dict[str, Any]:
    """Parse a Sentence element."""
    return {
        "num": sentence.get("Num"),
        "function": sentence.get("Function"),
        "writing_mode": sentence.get("WritingMode"),
        "text": sentence.text.strip() if sentence.text else ""
    }


def parse_paragraph(paragraph: ET.Element) -> Dict[str, Any]:
    """Parse a Paragraph element."""
    result = {
        "num": paragraph.get("Num"),
        "paragraph_num": parse_element_text(paragraph.find("ParagraphNum")),
        "caption": parse_element_text(paragraph.find("ParagraphCaption")),
        "sentences": [],
        "items": []
    }
    
    # Parse sentences
    for sentence_elem in paragraph.findall(".//ParagraphSentence/Sentence"):
        result["sentences"].append(parse_sentence(sentence_elem))
    
    # Parse items
    for item in paragraph.findall("Item"):
        item_data = {
            "num": item.get("Num"),
            "title": parse_element_text(item.find("ItemTitle")),
            "sentences": []
        }
        for sentence in item.findall(".//ItemSentence/Sentence"):
            item_data["sentences"].append(parse_sentence(sentence))
        for sentence in item.findall(".//ItemSentence/Column/Sentence"):
            item_data["sentences"].append(parse_sentence(sentence))
        result["items"].append(item_data)
    
    return result


def parse_article(article: ET.Element) -> Dict[str, Any]:
    """Parse an Article element."""
    result = {
        "num": article.get("Num"),
        "caption": parse_element_text(article.find("ArticleCaption")),
        "title": parse_element_text(article.find("ArticleTitle")),
        "paragraphs": []
    }
    
    for paragraph in article.findall("Paragraph"):
        result["paragraphs"].append(parse_paragraph(paragraph))
    
    return result


def parse_chapter(chapter: ET.Element) -> Dict[str, Any]:
    """Parse a Chapter element."""
    result = {
        "num": chapter.get("Num"),
        "title": parse_element_text(chapter.find("ChapterTitle")),
        "articles": []
    }
    
    for article in chapter.findall("Article"):
        result["articles"].append(parse_article(article))
    
    return result


def parse_toc(toc: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
    """Parse the Table of Contents (TOC)."""
    if toc is None:
        return None
    
    result = {
        "label": parse_element_text(toc.find("TOCLabel")),
        "chapters": []
    }
    
    for toc_chapter in toc.findall("TOCChapter"):
        chapter_data = {
            "num": toc_chapter.get("Num"),
            "title": parse_element_text(toc_chapter.find("ChapterTitle")),
            "article_range": parse_element_text(toc_chapter.find("ArticleRange"))
        }
        result["chapters"].append(chapter_data)
    
    # Parse supplementary provision in TOC
    toc_suppl = toc.find("TOCSupplProvision")
    if toc_suppl is not None:
        result["supplementary_provision_label"] = parse_element_text(
            toc_suppl.find("SupplProvisionLabel")
        )
    
    return result


def parse_supplementary_provision(suppl: ET.Element) -> Dict[str, Any]:
    """Parse a SupplProvision element."""
    result = {
        "amend_law_num": suppl.get("AmendLawNum"),
        "extract": suppl.get("Extract"),
        "label": parse_element_text(suppl.find("SupplProvisionLabel")),
        "articles": [],
        "paragraphs": []
    }
    
    for article in suppl.findall("Article"):
        result["articles"].append(parse_article(article))
    
    for paragraph in suppl.findall("Paragraph"):
        result["paragraphs"].append(parse_paragraph(paragraph))
    
    return result


def parse_law_full_text(law_full_text: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
    """Parse the full text of the law."""
    if law_full_text is None:
        return None
    
    law = law_full_text.find("Law")
    if law is None:
        return None
    
    result = {
        "attributes": {
            "era": law.get("Era"),
            "lang": law.get("Lang"),
            "law_type": law.get("LawType"),
            "num": law.get("Num"),
            "promulgate_day": law.get("PromulgateDay"),
            "promulgate_month": law.get("PromulgateMonth"),
            "year": law.get("Year")
        },
        "law_num": parse_element_text(law.find("LawNum")),
        "law_body": None
    }
    
    law_body = law.find("LawBody")
    if law_body is not None:
        law_title = law_body.find("LawTitle")
        result["law_body"] = {
            "title": {
                "text": law_title.text if law_title is not None else None,
                "abbrev": law_title.get("Abbrev") if law_title is not None else None,
                "abbrev_kana": law_title.get("AbbrevKana") if law_title is not None else None,
                "kana": law_title.get("Kana") if law_title is not None else None
            } if law_title is not None else None,
            "toc": parse_toc(law_body.find("TOC")),
            "main_provision": {
                "chapters": []
            },
            "supplementary_provisions": []
        }
        
        # Parse main provision
        main_provision = law_body.find("MainProvision")
        if main_provision is not None:
            for chapter in main_provision.findall("Chapter"):
                result["law_body"]["main_provision"]["chapters"].append(
                    parse_chapter(chapter)
                )
        
        # Parse supplementary provisions
        for suppl in law_body.findall("SupplProvision"):
            result["law_body"]["supplementary_provisions"].append(
                parse_supplementary_provision(suppl)
            )
    
    return result


def parse_law_info(law_info: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
    """Parse the law_info element."""
    if law_info is None:
        return None
    
    return {
        "law_type": parse_element_text(law_info.find("law_type")),
        "law_id": parse_element_text(law_info.find("law_id")),
        "law_num": parse_element_text(law_info.find("law_num")),
        "law_num_era": parse_element_text(law_info.find("law_num_era")),
        "law_num_year": parse_element_text(law_info.find("law_num_year")),
        "law_num_type": parse_element_text(law_info.find("law_num_type")),
        "law_num_num": parse_element_text(law_info.find("law_num_num")),
        "promulgation_date": parse_element_text(law_info.find("promulgation_date"))
    }


def parse_revision_info(revision_info: Optional[ET.Element]) -> Optional[Dict[str, Any]]:
    """Parse the revision_info element."""
    if revision_info is None:
        return None
    
    return {
        "law_revision_id": parse_element_text(revision_info.find("law_revision_id")),
        "law_type": parse_element_text(revision_info.find("law_type")),
        "law_title": parse_element_text(revision_info.find("law_title")),
        "law_title_kana": parse_element_text(revision_info.find("law_title_kana")),
        "abbrev": parse_element_text(revision_info.find("abbrev")),
        "category": parse_element_text(revision_info.find("category")),
        "updated": parse_element_text(revision_info.find("updated")),
        "amendment_promulgate_date": parse_element_text(revision_info.find("amendment_promulgate_date")),
        "amendment_enforcement_date": parse_element_text(revision_info.find("amendment_enforcement_date")),
        "amendment_enforcement_comment": parse_element_text(revision_info.find("amendment_enforcement_comment")),
        "amendment_scheduled_enforcement_date": parse_element_text(revision_info.find("amendment_scheduled_enforcement_date")),
        "amendment_law_id": parse_element_text(revision_info.find("amendment_law_id")),
        "amendment_law_title": parse_element_text(revision_info.find("amendment_law_title")),
        "amendment_law_title_kana": parse_element_text(revision_info.find("amendment_law_title_kana")),
        "amendment_law_num": parse_element_text(revision_info.find("amendment_law_num")),
        "amendment_type": parse_element_text(revision_info.find("amendment_type")),
        "repeal_status": parse_element_text(revision_info.find("repeal_status")),
        "repeal_date": parse_element_text(revision_info.find("repeal_date")),
        "remain_in_force": parse_element_text(revision_info.find("remain_in_force")),
        "mission": parse_element_text(revision_info.find("mission")),
        "current_revision_status": parse_element_text(revision_info.find("current_revision_status"))
    }


def parse_xml_file(xml_path: Path) -> Dict[str, Any]:
    """
    Parse a single XML file and return structured data.
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        Dictionary containing the parsed law data
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    result = {
        "source_file": xml_path.name,
        "parsed_at": datetime.now().isoformat(),
        "law_info": parse_law_info(root.find("law_info")),
        "revision_info": parse_revision_info(root.find("revision_info")),
        "law_full_text": parse_law_full_text(root.find("law_full_text")),
        "attached_files_info": None  # Usually empty, but preserved for completeness
    }
    
    # Check for attached files info
    attached = root.find("attached_files_info")
    if attached is not None and len(attached) > 0:
        result["attached_files_info"] = [
            {"name": child.tag, "value": child.text} for child in attached
        ]
    
    return result


def clean_none_values(obj: Any) -> Any:
    """
    Recursively remove None values and empty lists/dicts from the output
    to keep JSON clean and compact.
    """
    if isinstance(obj, dict):
        return {k: clean_none_values(v) for k, v in obj.items() 
                if v is not None and v != [] and v != {}}
    elif isinstance(obj, list):
        return [clean_none_values(item) for item in obj if item is not None]
    return obj


def process_all_xml_files(clean_output: bool = True) -> List[Dict[str, Any]]:
    """
    Process all XML files in the raw directory.
    
    Args:
        clean_output: If True, removes None values and empty structures
        
    Returns:
        List of parsed law data dictionaries
    """
    ensure_processed_dir()
    
    results = []
    xml_files = list(RAW_XML_DIR.glob("*.xml"))
    
    print(f"Found {len(xml_files)} XML files to process")
    
    for xml_file in xml_files:
        try:
            print(f"Processing: {xml_file.name}")
            data = parse_xml_file(xml_file)
            
            if clean_output:
                data = clean_none_values(data)
            
            # Save individual JSON file
            output_path = PROCESSED_DIR / f"{xml_file.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  -> Saved: {output_path.name}")
            results.append(data)
            
        except ET.ParseError as e:
            print(f"  -> ERROR parsing {xml_file.name}: {e}")
        except Exception as e:
            print(f"  -> ERROR processing {xml_file.name}: {e}")
    
    # Save combined JSON with all laws
    combined_output = PROCESSED_DIR / "_all_laws.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved combined file: {combined_output.name}")
    
    return results


def get_law_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of a law from parsed data.
    Useful for creating an index or quick lookup.
    """
    revision = data.get("revision_info", {}) or {}
    law_info = data.get("law_info", {}) or {}
    
    law_body = (data.get("law_full_text", {}) or {}).get("law_body", {}) or {}
    main_provision = law_body.get("main_provision", {}) or {}
    
    # Count chapters and articles
    chapters = main_provision.get("chapters", [])
    total_articles = sum(
        len(ch.get("articles", [])) for ch in chapters
    )
    
    return {
        "law_id": law_info.get("law_id"),
        "title": revision.get("law_title"),
        "title_kana": revision.get("law_title_kana"),
        "abbrev": revision.get("abbrev"),
        "category": revision.get("category"),
        "law_type": law_info.get("law_type"),
        "promulgation_date": law_info.get("promulgation_date"),
        "current_revision_status": revision.get("current_revision_status"),
        "amendment_enforcement_date": revision.get("amendment_enforcement_date"),
        "chapter_count": len(chapters),
        "article_count": total_articles
    }


def create_laws_index() -> List[Dict[str, Any]]:
    """
    Create an index file with summaries of all processed laws.
    """
    ensure_processed_dir()
    
    json_files = list(PROCESSED_DIR.glob("*.json"))
    summaries = []
    
    for json_file in json_files:
        if json_file.name.startswith("_"):  # Skip combined/index files
            continue
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = get_law_summary(data)
        summary["file"] = json_file.name
        summaries.append(summary)
    
    # Sort by law_id
    summaries.sort(key=lambda x: x.get("law_id", ""))
    
    # Save index
    index_path = PROCESSED_DIR / "_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    print(f"Created index with {len(summaries)} laws: {index_path.name}")
    return summaries


if __name__ == "__main__":
    print("=" * 60)
    print("Japanese Law XML Parser")
    print("=" * 60)
    print(f"Input directory:  {RAW_XML_DIR}")
    print(f"Output directory: {PROCESSED_DIR}")
    print("=" * 60)
    print()
    
    # Process all XML files
    results = process_all_xml_files()
    
    print()
    print("=" * 60)
    print("Creating index...")
    create_laws_index()
    
    print()
    print("=" * 60)
    print(f"Processing complete! Processed {len(results)} files.")
    print("=" * 60)
