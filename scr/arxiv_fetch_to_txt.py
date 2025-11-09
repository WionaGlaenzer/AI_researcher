#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

import arxiv
from arxiv import UnexpectedEmptyPageError
from pdfminer.high_level import extract_text
import requests
import certifi

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","of","for","to","in","on","at","by",
    "with","without","from","into","about","over","under","between","is","are","was","were",
    "be","been","being","this","that","these","those","it","its","as","we","you","they",
    "i","our","their","your","via","using","use"
}

def normalize_filename(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:120]

def extract_keywords(prompt: str):
    """
    Treat the entire prompt as a phrase AND also as tokenized keywords.
    Shell usually strips quotes, so we add the full phrase explicitly.
    """
    phrase = prompt.strip()
    tokens = re.findall(r"[A-Za-z0-9\-]+", prompt.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]

    seen = set()
    ordered = []
    for t in [phrase] + tokens:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def build_abs_query(keywords):
    """
    Builds '(abs:"<full phrase>") OR (abs:"w1" AND abs:"w2" ...)'.
    If only one token, it devolves to the phrase only.
    """
    if not keywords:
        return 'all:""'  # harmless placeholder to satisfy API
    phrase = keywords[0]
    parts = [f'abs:"{phrase}"']
    for k in keywords[1:]:
        parts.append(f'abs:"{k}"')
    if len(parts) == 1:
        return parts[0]
    return f'({parts[0]}) OR (' + " AND ".join(parts[1:]) + ")"

def arxiv_date_range_last_5y():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5*365)  # ~5 years
    fmt = "%Y%m%d%H%M"  # UTC, as expected by arXiv submittedDate
    return start.strftime(fmt), now.strftime(fmt)

def build_query(prompt_keywords):
    core = build_abs_query(prompt_keywords)
    start_str, end_str = arxiv_date_range_last_5y()
    # Parenthesize the OR-clause before the date filter to avoid precedence surprises
    return f"({core}) AND submittedDate:[{start_str} TO {end_str}]"

def download_pdf_via_requests(url: str, out_path: str, timeout=30):
    """Download using requests with certifi CA bundle (avoids local TLS store issues)."""
    r = requests.get(url, stream=True, timeout=timeout, verify=certifi.where())
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    parser = argparse.ArgumentParser(
        description="Fetch arXiv papers (last 5 years) matching abstract keywords and convert PDFs to .txt"
    )
    parser.add_argument("prompt", help="Free-form prompt, e.g. few-shot learning for time series")
    parser.add_argument("--outdir", default=None, help="Output folder (default: data/arxiv_output)")
    parser.add_argument("--max-results", type=int, default=10, help="How many to download (default: 10)")
    # Keep candidates to a single page to avoid the 'UnexpectedEmptyPageError' page jump
    parser.add_argument("--candidates", type=int, default=25,
                        help="How many results to scan before filtering (default: 25)")
    args = parser.parse_args()

    # Resolve output directory
    if args.outdir is None:
        # Get project root (parent of scr/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.outdir = os.path.join(project_root, "data", "arxiv_output")
    elif not os.path.isabs(args.outdir):
        # If relative path, make it relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.outdir = os.path.join(project_root, args.outdir)

    keywords = extract_keywords(args.prompt)
    query = build_query(keywords)

    print(f"Keywords: {keywords}")
    print(f"arXiv query: {query}")

    out_pdf = os.path.join(args.outdir, "pdf")
    out_txt = os.path.join(args.outdir, "txt")
    os.makedirs(out_pdf, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)

    # Conservative client settings; page_size=25 keeps to a single page
    client = arxiv.Client(page_size=25, delay_seconds=3, num_retries=5)

    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.Relevance,
        max_results=max(1, min(args.candidates, 25))  # single page to dodge empty-page bug
    )

    try:
        results = list(client.results(search))
    except UnexpectedEmptyPageError:
        # Fallback: smaller pull if server hiccups
        search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.Relevance, max_results=10)
        results = list(client.results(search))

    # Keep top N
    results = results[: args.max_results]

    if not results:
        print("No matching results in the last 5 years.")
        sys.exit(0)

    print(f"Downloading {len(results)} paper(s)...")

    for i, r in enumerate(results, 1):
        arxiv_id = r.get_short_id()
        title = r.title or arxiv_id
        base = f"{arxiv_id}_{normalize_filename(title)}"

        pdf_path = os.path.join(out_pdf, f"{base}.pdf")
        txt_path = os.path.join(out_txt, f"{base}.txt")

        # Use requests+certifi for robust TLS verification
        try:
            pdf_url = r.pdf_url  # direct https link
            download_pdf_via_requests(pdf_url, pdf_path)
            print(f"[{i}/{len(results)}] Downloaded: {pdf_path}")
        except Exception as e:
            print(f"[{i}/{len(results)}] Failed to download PDF for {arxiv_id}: {e}")
            continue

        # Convert to TXT
        try:
            # Suppress pdfminer warnings about invalid color values
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    text = extract_text(pdf_path)
                finally:
                    sys.stderr = old_stderr
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Converted to TXT: {txt_path}")
        except Exception as e:
            print(f"Failed to convert {arxiv_id} to text: {e}")

    print("\nDone. PDFs in:", out_pdf)
    print("TXT files in:", out_txt)

if __name__ == "__main__":
    main()
