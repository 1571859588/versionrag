#!/usr/bin/env python3
"""
run_eda_benchmark.py — Non-interactive VersionRAG adapter for the LLM4EDA benchmark.

This script bridges the official VersionRAG pipeline to our benchmark:
  1. Index phase:  Feed v1.md and v2.md into VersionRAGIndexer
  2. Query phase:  Run each QA from qa_pairs.json through VersionRAGRetriever + VersionRAGGenerator
  3. Evolution:    Generate an evolution summary for ERS evaluation
  4. Output:       Save eval-compatible JSON to results/versionRAG/

Input files:
  - v1.md / v2.md : Markdown documents (VersionRAG supports both .md and .pdf natively)
  - benchmark/gts/qa_pairs.json
  - benchmark/gts/evolution_rationale.json

Usage:
  cd baselines/versionrag/src
  python run_eda_benchmark.py \
    --v1-text ../../../benchmark/test_markdown/v1.md \
    --v2-text ../../../benchmark/test_markdown/v2.md \
    --gt-dir ../../../benchmark/gts \
    --output-dir ../../../results/versionRAG

Environment (set via .env in src/ or as shell env vars):
  OPENAI_API_KEY      — required for embedding + LLM
  OPENAI_BASE_URL     — optional, for OpenAI-compatible endpoints
  NEO4J_URI           — Neo4j connection URI
  NEO4J_USER          — Neo4j username
  NEO4J_PASSWORD      — Neo4j password
"""

import sys
import os
import json
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
import copy
import socket

# ---------------------------------------------------------------------------
# Path setup: must run from baselines/versionrag/src/ or we add src to path
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROJECT_SRC = os.path.abspath(os.path.join(SRC_DIR, "../../../src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Load .env from src/
from dotenv import load_dotenv
dotenv_path = os.path.join(SRC_DIR, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"[ENV] Loaded .env from {dotenv_path}")
else:
    load_dotenv()  # fallback to shell env
    print(f"[ENV] .env not found at {dotenv_path}, using shell environment")

try:
    from config.prompts import get_baseline_generation_append
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="VersionRAG EDA Benchmark — non-interactive adapter"
    )
    parser.add_argument("--v1-text", required=True,
                        help="Path to v1 Markdown or PDF file")
    parser.add_argument("--v2-text", required=True,
                        help="Path to v2 Markdown or PDF file")
    parser.add_argument("--gt-dir", required=True,
                        help="Path to ground truth directory (qa_pairs.json etc.)")
    parser.add_argument("--output-dir", default="../../../results/versionRAG",
                        help="Output directory for results JSON")
    parser.add_argument("--domain", default="tcl",
                        help="Target script domain/language for code generation (e.g., tcl)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop existing Milvus collection and re-index")
    parser.add_argument("--query-only", action="store_true",
                        help="Skip checking/indexing, only run queries against existing Milvus collection.")
    return parser.parse_args()


def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"versionrag_run_{timestamp}.log")

    logger = logging.getLogger("versionrag_benchmark")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path, timestamp


def clear_milvus_collection(collection_name: str):
    """Drop existing Milvus collection for clean re-index."""
    from util.constants import MILVUS_DB_PATH
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(MILVUS_DB_PATH)
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
            print(f"  Dropped existing Milvus collection: {collection_name}")
        client.close()
    except Exception as e:
        print(f"  Warning: could not drop Milvus collection: {e}")


def clear_knowledge_graph():
    """Clear the VersionRAG knowledge graph pickle."""
    from util.constants import KNOWLEDGE_GRAPH_PATH
    kg_path = os.path.join(SRC_DIR, KNOWLEDGE_GRAPH_PATH)
    if os.path.exists(kg_path):
        os.remove(kg_path)
        print(f"  Removed existing knowledge graph: {kg_path}")


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    logger, log_path, timestamp = setup_logger(output_dir)
    log = logger.info

    log("=" * 70)
    log(f"VersionRAG EDA Benchmark Adapter | {datetime.now().isoformat()}")
    log("=" * 70)

    # Validate inputs
    for path, name in [(args.v1_text, "v1"), (args.v2_text, "v2"), (args.gt_dir, "gt-dir")]:
        abs_p = os.path.abspath(path)
        if not os.path.exists(abs_p):
            log(f"ERROR: {name} path not found: {abs_p}")
            sys.exit(1)

    v1_path = os.path.abspath(args.v1_text)
    v2_path = os.path.abspath(args.v2_text)
    gt_dir = os.path.abspath(args.gt_dir)

    log(f"\n[0] Inputs:")
    log(f"  V1:     {v1_path}")
    log(f"  V2:     {v2_path}")
    log(f"  GT dir: {gt_dir}")
    log(f"  Output: {output_dir}")
    log(f"  Rebuild: {args.rebuild}")
    log(f"  Domain: {args.domain}")
    log(f"  Query-only: {args.query_only}")

    # ------------------------------------------------------------------
    # 1. Load Ground Truth
    # ------------------------------------------------------------------
    log("\n[1] Loading Ground Truth...")
    qa_path = os.path.join(gt_dir, "qa_pairs.json")
    rationale_path = os.path.join(gt_dir, "evolution_rationale.json")

    with open(qa_path, "r", encoding="utf-8") as f:
        gt_qa = json.load(f)
    with open(rationale_path, "r", encoding="utf-8") as f:
        gt_rationale = json.load(f)

    log(f"  QA pairs:   {len(gt_qa['qa_pairs'])}")
    log(f"  Rationales: {len(gt_rationale['rationales'])}")

    # ------------------------------------------------------------------
    # 2. Indexing Phase  (VersionRAG pipeline)
    # ------------------------------------------------------------------
    log("\n[2] Indexing Phase (VersionRAGIndexer)...")

    from util.constants import MILVUS_COLLECTION_NAME_VERSIONRAG

    if args.rebuild:
        log("  --rebuild flag: clearing existing index and graph...")
        clear_milvus_collection(MILVUS_COLLECTION_NAME_VERSIONRAG)
        clear_knowledge_graph()

    # Check if index already exists (skip re-indexing)
    from util.constants import MILVUS_DB_PATH, KNOWLEDGE_GRAPH_PATH
    kg_path = os.path.join(SRC_DIR, KNOWLEDGE_GRAPH_PATH)
    milvus_path = os.path.join(SRC_DIR, os.path.dirname(MILVUS_DB_PATH))

    if os.path.exists(kg_path) and not args.rebuild and not args.query_only:
        log("  -> Existing index detected (skip re-indexing). Use --rebuild to force.")
    elif args.query_only:
        log("  --query-only flag: skipping indexing phase.")
    else:
        log("  -> Running VersionRAGIndexer.index_data()...")
        from indexing.versionrag_indexer import VersionRAGIndexer

        indexer = VersionRAGIndexer()
        data_files = [v1_path, v2_path]
        log(f"  -> Files: {[os.path.basename(p) for p in data_files]}")

        try:
            indexer.index_data(data_files)
            log("  -> Indexing complete!")
        except Exception as e:
            log(f"  ERROR during indexing: {e}")
            import traceback
            log(traceback.format_exc())
            sys.exit(1)

    # ------------------------------------------------------------------
    # 3. QA Query Phase
    # ------------------------------------------------------------------
    log(f"\n[3] Query Phase ({len(gt_qa['qa_pairs'])} questions)...")

    from retrieval.versionrag_retriever import VersionRAGRetriever
    from generation.versionrag_generator import VersionRAGGenerator

    try:
        retriever = VersionRAGRetriever()
        generator = VersionRAGGenerator()
    except Exception as e:
        log(f"  ERROR initializing retriever/generator: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

    qa_results = []
    appender = get_baseline_generation_append(args.domain) if 'get_baseline_generation_append' in globals() else ""
    for idx, qa in enumerate(gt_qa["qa_pairs"]):
        query = qa["query"]
        deprecated = qa.get("deprecated_terms_in_context", [])
        query_id = qa.get("query_id", f"query_{idx}")
        expected_rationale = qa.get("expected_rationale", "")
        expected_command = qa.get("expected_command", "")

        log(f"\n  [Q{idx+1}]: {query}")

        try:
            retrieved_data = retriever.retrieve(query)
            response = generator.generate(retrieved_data, query + appender) # Apply appender here
            response_text = response.answer if hasattr(response, "answer") else str(response)
            log(f"  [A{idx+1}]: {str(response_text)[:400]}")
        except Exception as e:
            log(f"  [Error Q{idx+1}]: {e}")
            response_text = f"[VersionRAG Error]: {e}"

        qa_results.append({
            "query_id": query_id,
            "query": query,
            "expected_rationale": expected_rationale,
            "expected_command": expected_command,
            "response": str(response_text).strip(),
            "deprecated_terms": deprecated,
        })

    # ------------------------------------------------------------------
    # 4. ERS — Generate evolution rationale
    # ------------------------------------------------------------------
    log("\n[4] Generating evolution rationale (ERS)...")
    evo_query = (
        "Summarize all changes and evolution from version 1.0 to version 2.0 "
        "of this documentation. List each specific change as a bullet point, "
        "including renamed commands, new parameters, and deprecated features."
    )

    try:
        evo_retrieved = retriever.retrieve(evo_query)
        evo_response = generator.generate(evo_retrieved, evo_query)
        evo_rationale = evo_response.answer if hasattr(evo_response, "answer") else str(evo_response)
        log(f"  [ERS]: {str(evo_rationale)[:500]}")
    except Exception as e:
        log(f"  [ERS Error]: {e}")
        evo_rationale = f"[VersionRAG Error]: {e}"

    # ------------------------------------------------------------------
    # 5. Save output JSON (eval-compatible format)
    # ------------------------------------------------------------------
    log("\n[5] Saving results...")

    output = {
        "baseline": "VersionRAG",
        "timestamp": timestamp,
        "config": {
            "v1_input": v1_path,
            "v2_input": v2_path,
            "input_type": "markdown" if v1_path.lower().endswith(".md") else "pdf",
        },
        "qa_results": qa_results,
        "ers_rationale": str(evo_rationale).strip(),
        "gt_rationale_text": "\n".join(gt_rationale["rationales"]),
    }

    result_filename = f"versionrag_result_{timestamp}.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"\n  Log:    {log_path}")
    log(f"  Result: {result_path}")
    log("\n" + "=" * 70)
    log("VersionRAG run complete! Use eval_baselines.py to compute DHR/ERS scores.")
    log("=" * 70)


if __name__ == "__main__":
    main()
