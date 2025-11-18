"""
Step 06: Label dataset into 6 task-specific categories
=======================================================
- Fetches all records from Supabase modelComp
- Classifies each input into one of 6 categories
- Updates label column in Supabase

Categories:
1. technical_code - Programming, algorithms, APIs, data structures
2. math_logic - Calculations, sequences, logical reasoning
3. classification_analysis - Categorize, analyze tone/sentiment, identify
4. language_editing - Rewrite, summarize, translate, grammar
5. creative_generative - Stories, poems, marketing content, recipes
6. general_qa - Factual info, definitions, explanations (default)

Usage:
    python experiment/06_label_dataset.py
    python experiment/06_label_dataset.py --dry-run  # Preview without updating
"""

import re
import os
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Category rules (checked in priority order)
# Higher priority = more specific categories first
RULES = {
    "technical_code": [
        "write code", "write a code", "code snippet", "source code",
        "function", "algorithm", "regex", "regular expression",
        "sql query", "database", "programming", " api ", "api call",
        "python", "java ", "javascript", "html", "css", "c++",
        "debug", "compile", "syntax error", "variable", "class ",
        "method", "for loop", "while loop", "array", "data structure",
        "binary tree", "linked list", "hash table", "stack", "queue",
        "recursion", "sorting", "bubble sort", "merge sort",
        "implement", "script", "terminal", "command line", "git",
        "docker", "server", "http request", "json", "xml", "parse",
        "encode", "decode", "encrypt", "decrypt", "authentication",
        "backend", "frontend", "framework", "library", "module",
        "import ", "def ", "return ", "print(", "console.log",
    ],
    "math_logic": [
        "calculate", "compute", "area of", "median", "average",
        "sum of", "equation", "probability", "percentage",
        "fraction", "decimal", "multiply", "divide", "subtract",
        "algebra", "geometry", "calculus", "derivative", "integral",
        "sequence", "series", "pattern", "formula", "solve for",
        "proof", "theorem", "hypothesis", "deduce", "infer",
        "mathematical", "arithmetic", "statistic", "mean of",
        "standard deviation", "variance", "factorial", "permutation",
        "combination", "prime number", "odd number", "even number",
        "greater than", "less than", "equal to", "how many",
        "what is the value", "find the", "the answer is",
    ],
    "classification_analysis": [
        "classify", "identify", "tone", "sentiment", "categorize",
        "odd one out", "label", "theme", "analyze", "analysis",
        "compare", "contrast", "differentiate", "distinguish",
        "evaluate", "assess", "rate", "rank", "prioritize",
        "pros and cons", "advantages", "disadvantages", "critique",
        "review", "examine", "inspect", "determine", "which of",
        "sort into", "group", "cluster", "organize", "order",
    ],
    "language_editing": [
        "rewrite", "summarize", "summary", "translate", "edit",
        "concise", "active voice", "passive voice", "grammar",
        "paraphrase", "rephrase", "simplify", "shorten", "condense",
        "proofread", "correct", "fix the", "improve", "revise",
        "formal", "informal", "professional", "casual", "brief",
        "tl;dr", "key points", "main idea", "outline", "synopsis",
        "convert to", "change to", "make it", "transform",
    ],
    "creative_generative": [
        "write a story", "poem", "haiku", "slogan", "tweet", "post",
        "recipe", "pitch", "creative", "compose", "create a", "generate",
        "imagine", "fiction", "narrative", "character", "dialogue",
        "script", "screenplay", "lyrics", "song", "jingle", "tagline",
        "advertisement", "marketing", "brand", "catchy", "engaging",
        "persuasive", "inspiring", "motivational", "emotional",
        "humorous", "funny", "joke", "riddle", "puzzle", "game",
        "roleplay", "scenario", "hypothetical", "fantasy", "dream",
        "invent", "design", "brainstorm", "idea", "concept",
    ],
}


def classify_input(instruction: str, input_text: str = "") -> str:
    """
    Classify based on instruction + input combined.
    Returns category name based on priority order.
    """
    combined = f"{instruction} {input_text}".lower()
    
    if not combined.strip():
        return "general_qa"
    
    # Check each category in priority order
    priority_order = [
        "technical_code",      # Most specific
        "math_logic",          
        "classification_analysis",
        "language_editing",
        "creative_generative",
    ]
    
    for category in priority_order:
        keywords = RULES[category]
        for keyword in keywords:
            if keyword in combined:
                return category
    
    # Default fallback
    return "general_qa"


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def fetch_all_records(supabase):
    """Fetch ALL records with pagination"""
    print("Fetching records from Supabase...")
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("modelComp")\
            .select("id, input")\
            .range(offset, offset + page_size - 1)\
            .execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched page {offset // page_size + 1}: {len(response.data)} records")
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    print(f"✓ Total records: {len(all_records)}")
    return all_records


def update_label(supabase, record_id: str, label: str):
    """Update label column for a record"""
    try:
        supabase.table("modelComp")\
            .update({"label": label})\
            .eq("id", record_id)\
            .execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Update error for {record_id}: {e}")
        return False


def main(dry_run: bool = False):
    """Label all records in Supabase"""
    
    print("=" * 60)
    print("STEP 06: LABEL DATASET INTO CATEGORIES")
    print("=" * 60)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No updates will be made\n")
    
    # Connect to Supabase
    supabase = get_supabase_client()
    print("✓ Connected to Supabase\n")
    
    # Fetch all records
    records = fetch_all_records(supabase)
    
    if not records:
        print("✗ No records found!")
        return
    
    print(f"\nClassifying {len(records)} records...")
    print("-" * 60)
    
    # Classify and track distribution
    category_counts = Counter()
    results = []
    
    for record in tqdm(records, desc="Classifying"):
        input_text = record.get("input", "")
        label = classify_input(input_text, "")  # instruction is in input field
        category_counts[label] += 1
        results.append((record["id"], label))
    
    # Show distribution
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION")
    print("=" * 60)
    
    total = len(records)
    categories = [
        "technical_code", 
        "math_logic", 
        "classification_analysis", 
        "language_editing", 
        "creative_generative", 
        "general_qa"
    ]
    for category in categories:
        count = category_counts[category]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {category:25} {count:5} ({pct:5.1f}%) {bar}")
    
    print("-" * 60)
    print(f"  {'TOTAL':25} {total:5}")
    
    if dry_run:
        print("\n🔍 DRY RUN - No updates made")
        print("   Run without --dry-run to update Supabase")
        
        # Show some examples
        print("\n📋 Sample classifications:")
        for category in categories[:4]:
            samples = [(r["input"][:70], l) for r, (rid, l) in zip(records, results) if l == category][:2]
            if samples:
                print(f"\n  [{category}]")
                for text, _ in samples:
                    print(f"    • {text}...")
        return
    
    # Update Supabase
    print(f"\nUpdating Supabase labels...")
    success = 0
    failed = 0
    
    for record_id, label in tqdm(results, desc="Updating labels"):
        if update_label(supabase, record_id, label):
            success += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)
    print(f"✓ Updated: {success}")
    print(f"✗ Failed: {failed}")
    print("=" * 60)
    print("\n→ Next: Define metrics per category and evaluate")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", 
                       help="Preview classification without updating Supabase")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
