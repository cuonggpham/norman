#!/usr/bin/env python3
"""
e-Gov Laws API Downloader

Downloads Japanese legal data for RAG system supporting financial consulting for foreigners.

Usage:
    python scripts/downloader.py                    # Download from existing law_ids.txt
    python scripts/downloader.py --search           # Search & add new law IDs for finance
    python scripts/downloader.py --keyword 外国人   # Search by keyword
    python scripts/downloader.py --category 国税    # Search by category
    python scripts/downloader.py --list-categories  # List available categories
"""

import requests
import os
import time
import argparse
from typing import Optional, List, Set
from pathlib import Path

# Configuration
BASE_URL = "https://laws.e-gov.go.jp/api/2"
# Paths relative to project root (/home/dell/Documents/Code/norman)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts -> backend -> norman
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "xml_raw")
INPUT_FILE = str(PROJECT_ROOT / "data" / "law_ids.txt")
RATE_LIMIT_DELAY = 1.2  # seconds

# Categories related to financial consulting for foreigners
FINANCIAL_CATEGORIES = [
    "国税",           # National Tax
    "地方財政",       # Local Finance  
    "社会保険",       # Social Insurance
    "労働",           # Labor
]

# Keywords for foreign residents
FOREIGNER_KEYWORDS = [
    "外国人",         # Foreigner
    "在留",           # Residence
    "入国",           # Immigration
    "所得税",         # Income tax
    "住民税",         # Resident tax
    "年金",           # Pension
    "健康保険",       # Health insurance
]

# Filter settings for RAG quality
MODERN_ERAS = ["Showa", "Heisei", "Reiwa"]  # 1926+, skip Meiji/Taisho
PRIORITY_LAW_TYPES = ["Act"]  # Primary focus on Acts
ALLOWED_LAW_TYPES = ["Act", "CabinetOrder"]  # Allow Acts + Cabinet Orders
VALID_STATUS = ["CurrentEnforced"]  # Only currently enforced laws


def load_existing_ids() -> Set[str]:
    """Load existing law IDs from file."""
    if not os.path.exists(INPUT_FILE):
        return set()
    
    with open(INPUT_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_ids(ids: Set[str]):
    """Save law IDs to file."""
    with open(INPUT_FILE, "w") as f:
        for law_id in sorted(ids):
            f.write(f"{law_id}\n")


def get_laws_list(
    category: Optional[str] = None,
    law_type: Optional[str] = None,
    promulgation_date_from: Optional[str] = None,
    updated_from: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[dict]:
    """
    GET /laws - Legal List Acquisition API
    
    Fetches list of laws with optional filters.
    """
    url = f"{BASE_URL}/laws"
    params = {
        "limit": min(limit, 500),  # API max is 500
        "offset": offset,
    }
    
    if category:
        params["category"] = category
    if law_type:
        params["law_type"] = law_type
    if promulgation_date_from:
        params["promulgation_date_from"] = promulgation_date_from
    if updated_from:
        params["updated_from"] = updated_from
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # API returns nested structure: laws[].law_info.law_id
        laws = data.get("laws", [])
        # Flatten to extract law_id for easier processing
        result = []
        for law in laws:
            law_info = law.get("law_info", {})
            revision_info = law.get("revision_info", {})
            result.append({
                "law_id": law_info.get("law_id"),
                "law_title": revision_info.get("law_title"),
                "category": revision_info.get("category"),
                "era": law_info.get("law_num_era"),
                "law_type": law_info.get("law_type"),
                "current_status": revision_info.get("current_revision_status"),
                "repeal_status": revision_info.get("repeal_status"),
            })
        return result
    except requests.RequestException as e:
        print(f"❌ Error fetching laws list: {e}")
        return []


def search_keyword(
    keyword: str,
    law_type: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[dict]:
    """
    GET /keyword - Keyword Search API
    
    Searches laws by keyword.
    """
    url = f"{BASE_URL}/keyword"
    params = {
        "keyword": keyword,
        "limit": min(limit, 500),
        "offset": offset,
    }
    
    if law_type:
        params["law_type"] = law_type
    if category:
        params["category"] = category
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Keyword API returns 'items' instead of 'laws'
        items = data.get("items", [])
        result = []
        for item in items:
            law_info = item.get("law_info", {})
            revision_info = item.get("revision_info", {})
            result.append({
                "law_id": law_info.get("law_id"),
                "law_title": revision_info.get("law_title"),
                "category": revision_info.get("category"),
                "era": law_info.get("law_num_era"),
                "law_type": law_info.get("law_type"),
                "current_status": revision_info.get("current_revision_status"),
                "repeal_status": revision_info.get("repeal_status"),
            })
        return result
    except requests.RequestException as e:
        print(f"❌ Error searching keyword '{keyword}': {e}")
        return []


def get_law_revisions(law_id_or_num: str) -> List[dict]:
    """
    GET /law_revisions/{law_id_or_num} - Legal History List Acquisition API
    
    Gets revision history for a specific law.
    """
    url = f"{BASE_URL}/law_revisions/{law_id_or_num}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("law_revisions", [])
    except requests.RequestException as e:
        print(f"❌ Error fetching revisions for {law_id_or_num}: {e}")
        return []


def download_law_data(law_id: str, output_dir: str = OUTPUT_DIR) -> bool:
    """
    GET /law_data/{law_id} - Get the text of the law API
    
    Downloads law data in XML format.
    """
    url = f"{BASE_URL}/law_data/{law_id}"
    headers = {"Accept": "application/xml"}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f"{law_id}.xml")
            with open(file_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"❌ Error {law_id}: Status {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"⚠️ Connection error {law_id}: {e}")
        return False


def filter_laws(
    laws: List[dict], 
    modern_only: bool = True,
    acts_only: bool = False,
    current_only: bool = True
) -> List[dict]:
    """
    Filter laws based on RAG quality criteria.
    
    Args:
        laws: List of law dicts with era, law_type, current_status fields
        modern_only: Only Showa/Heisei/Reiwa laws (1926+)
        acts_only: Only Acts (exclude ordinances/circulars)
        current_only: Only CurrentEnforced laws
    """
    filtered = []
    stats = {"era": 0, "type": 0, "status": 0}
    
    for law in laws:
        # Filter by era (skip Meiji/Taisho)
        if modern_only and law.get("era") not in MODERN_ERAS:
            stats["era"] += 1
            continue
        
        # Filter by law type
        if acts_only and law.get("law_type") not in PRIORITY_LAW_TYPES:
            stats["type"] += 1
            continue
        elif not acts_only and law.get("law_type") not in ALLOWED_LAW_TYPES:
            stats["type"] += 1
            continue
        
        # Filter by current status
        if current_only and law.get("current_status") not in VALID_STATUS:
            stats["status"] += 1
            continue
        
        filtered.append(law)
    
    print(f"  📊 Filtered: era={stats['era']}, type={stats['type']}, status={stats['status']}")
    return filtered


def search_financial_laws(apply_filters: bool = True, acts_only: bool = False) -> Set[str]:
    """
    Search for laws related to financial consulting for foreigners.
    
    Args:
        apply_filters: Apply quality filters (modern era, current status)
        acts_only: Only include Acts (skip ordinances)
    
    Returns set of law IDs.
    """
    all_laws = []
    
    # Search by financial categories
    print("🔍 Searching by financial categories...")
    for category in FINANCIAL_CATEGORIES:
        print(f"  📂 Category: {category}")
        laws = get_laws_list(category=category, limit=500)
        all_laws.extend(laws)
        print(f"     Found {len(laws)} laws")
        time.sleep(RATE_LIMIT_DELAY)
    
    # Search by foreigner-related keywords  
    print("\n🔍 Searching by keywords...")
    for keyword in FOREIGNER_KEYWORDS:
        print(f"  🔑 Keyword: {keyword}")
        laws = search_keyword(keyword=keyword, limit=200)
        all_laws.extend(laws)
        print(f"     Found {len(laws)} laws")
        time.sleep(RATE_LIMIT_DELAY)
    
    # Deduplicate
    seen = set()
    unique_laws = []
    for law in all_laws:
        law_id = law.get("law_id")
        if law_id and law_id not in seen:
            seen.add(law_id)
            unique_laws.append(law)
    
    print(f"\n📋 Total unique laws found: {len(unique_laws)}")
    
    # Apply filters if requested
    if apply_filters:
        print("\n🔧 Applying quality filters...")
        filtered = filter_laws(unique_laws, modern_only=True, acts_only=acts_only, current_only=True)
        print(f"✅ After filtering: {len(filtered)} laws")
        return {law.get("law_id") for law in filtered if law.get("law_id")}
    
    return {law.get("law_id") for law in unique_laws if law.get("law_id")}


def download_all(ids: Set[str], output_dir: str = OUTPUT_DIR):
    """Download all laws from the given IDs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which ones already exist
    existing_files = {f.replace(".xml", "") for f in os.listdir(output_dir) if f.endswith(".xml")}
    to_download = ids - existing_files
    
    if not to_download:
        print("✅ All laws already downloaded!")
        return
    
    print(f"\n📥 Downloading {len(to_download)} laws (skipping {len(existing_files)} existing)...")
    
    for idx, law_id in enumerate(sorted(to_download), 1):
        success = download_law_data(law_id, output_dir)
        status = "✅" if success else "❌"
        print(f"  {status} [{idx}/{len(to_download)}] {law_id}")
        time.sleep(RATE_LIMIT_DELAY)
    
    print("\n🎉 Download complete!")


def list_categories():
    """List commonly used law categories."""
    categories = [
        ("国税", "National Tax"),
        ("地方財政", "Local Finance"),
        ("社会保険", "Social Insurance"),
        ("労働", "Labor"),
        ("出入国管理", "Immigration Control"),
        ("通商", "Trade"),
        ("金融", "Finance"),
        ("商業", "Commerce"),
        ("会社", "Companies"),
    ]
    
    print("\n📋 Available Categories (commonly used):\n")
    for jp, en in categories:
        print(f"  {jp:<12} - {en}")
    print("\nNote: These are common categories. The API may have more.")


def main():
    parser = argparse.ArgumentParser(
        description="e-Gov Laws API Downloader for RAG system"
    )
    parser.add_argument(
        "--search", 
        action="store_true",
        help="Search for new financial laws and add to law_ids.txt"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="Search laws by specific keyword"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Search laws by category (e.g., 国税, 労働)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results per search (default: 100)"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, don't search for new IDs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually downloading"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable quality filters (include old laws, ordinances, etc.)"
    )
    parser.add_argument(
        "--acts-only",
        action="store_true",
        help="Only include Acts (skip Cabinet Orders and Ministerial Ordinances)"
    )
    
    args = parser.parse_args()
    
    # List categories
    if args.list_categories:
        list_categories()
        return
    
    # Load existing IDs
    existing_ids = load_existing_ids()
    print(f"📄 Loaded {len(existing_ids)} existing law IDs from {INPUT_FILE}")
    
    new_ids = set()
    apply_filters = not args.no_filter
    
    # Search by keyword
    if args.keyword:
        print(f"\n🔍 Searching for keyword: {args.keyword}")
        laws = search_keyword(keyword=args.keyword, limit=args.limit)
        if apply_filters:
            print("\n🔧 Applying quality filters...")
            laws = filter_laws(laws, modern_only=True, acts_only=args.acts_only, current_only=True)
        new_ids = {law.get("law_id") for law in laws if law.get("law_id")}
        print(f"   Found {len(new_ids)} laws")
    
    # Search by category
    elif args.category:
        print(f"\n🔍 Searching for category: {args.category}")
        laws = get_laws_list(category=args.category, limit=args.limit)
        if apply_filters:
            print("\n🔧 Applying quality filters...")
            laws = filter_laws(laws, modern_only=True, acts_only=args.acts_only, current_only=True)
        new_ids = {law.get("law_id") for law in laws if law.get("law_id")}
        print(f"   Found {len(new_ids)} laws")
    
    # Full financial search
    elif args.search:
        new_ids = search_financial_laws(apply_filters=apply_filters, acts_only=args.acts_only)
    
    # Add new IDs
    if new_ids:
        truly_new = new_ids - existing_ids
        if truly_new:
            print(f"\n✨ Found {len(truly_new)} new law IDs")
            all_ids = existing_ids | new_ids
            
            if args.dry_run:
                print("   (Dry run - not saving)")
                for law_id in sorted(truly_new)[:10]:
                    print(f"     - {law_id}")
                if len(truly_new) > 10:
                    print(f"     ... and {len(truly_new) - 10} more")
            else:
                save_ids(all_ids)
                print(f"   Saved to {INPUT_FILE}")
                existing_ids = all_ids
        else:
            print("\n📋 No new law IDs found (all already exist)")
    
    # Download
    if not args.dry_run and not args.search and not args.keyword and not args.category:
        # Default behavior: download from existing IDs
        download_all(existing_ids)
    elif not args.dry_run and not args.download_only and (args.search or args.keyword or args.category):
        # After search, ask if want to download
        if new_ids:
            download_all(existing_ids | new_ids)


if __name__ == "__main__":
    main()