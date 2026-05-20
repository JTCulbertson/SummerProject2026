"""
scripts/judge/analyze.py — LLM Judge: Single-Puzzle Metaphor Analysis
======================================================================
Runs the LLM-as-judge pipeline on all unprocessed marc_runs for ONE specific
puzzle ID. Reads from the source DB (data/gpt5.db) and writes judge verdicts
(was_metaphor_used, confidence_score, judge_reasoning) to a companion analysis DB.

Usage:
    python scripts/judge/analyze.py <puzzle_id> [--db gpt5.db]

Output:
    data/<source>_analysis.db  — appended with new metaphor_analysis rows

Related:
    analyze_all.py  — same pipeline, processes every puzzle in the DB at once
    report.py       — generates an HTML report from the populated analysis DB
"""
import json
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import (
    Boolean, Column, Enum, Float, ForeignKey, Integer,
    String, Text, DateTime, create_engine, select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, joinedload
from sqlalchemy.exc import OperationalError

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent   # scripts/judge/ -> scripts/ -> project root

DEFAULT_DB = BASE_DIR / "data" / "gpt5.db"
CONNECT_SECS   = 10
LONG_READ_SECS = 60

# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

Base = declarative_base()

class Puzzle(Base):
    __tablename__ = "puzzles"
    id         = Column(Integer, primary_key=True)
    discipline = Column(String, nullable=False)
    arc_type   = Column(Enum("ARC", "ARC2", "MARC", name="arc_types"), nullable=False)
    metaphor   = Column(String, nullable=True)
    file_path  = Column(String, nullable=False)
    runs       = relationship("MarcRun", back_populates="puzzle", cascade="all, delete-orphan")

class MarcRun(Base):
    __tablename__ = "marc_runs"
    id                  = Column(Integer, primary_key=True)
    puzzle_id           = Column(Integer, ForeignKey("puzzles.id"), nullable=False)
    did_pass            = Column(Boolean, nullable=False)
    time_to_completion  = Column(Float, nullable=False)
    num_reasoning_chars = Column(Integer, nullable=False)
    model               = Column(String, nullable=False)
    coding_type         = Column(String, nullable=False)
    instruct_type       = Column(String, nullable=False)
    reasoning_text      = Column(Text, nullable=True)
    puzzle              = relationship("Puzzle", back_populates="runs")

AnalysisBase = declarative_base()

class MetaphorAnalysis(AnalysisBase):
    __tablename__ = "metaphor_analysis"
    id                 = Column(Integer, primary_key=True)
    marc_run_id        = Column(Integer, nullable=False, unique=True)
    judge_model        = Column(String, nullable=False)
    was_metaphor_used  = Column(Boolean, nullable=False)
    confidence_score   = Column(Float, nullable=False)
    judge_reasoning    = Column(Text, nullable=True)
    analysis_timestamp = Column(DateTime, default=datetime.now)
    coding_type        = Column(String, nullable=True)
    instruct_type      = Column(String, nullable=True)

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _make_http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.8,
                  status_forcelist=(502, 503, 504),
                  allowed_methods=frozenset(["POST", "GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

def _post(url, headers, payload):
    return _make_http_session().post(
        url, headers=headers, json=payload,
        timeout=(CONNECT_SECS, LONG_READ_SECS), stream=False
    )

def _parse_json_lenient(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", s, flags=re.DOTALL)
        return json.loads(s)

def _pick_content(data: Any) -> Optional[str]:
    if not isinstance(data, dict): return str(data)
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        if "content" in msg: return msg["content"]
    if "message" in data and isinstance(data["message"], dict):
        if "content" in data["message"]: return data["message"]["content"]
    if "content" in data and isinstance(data["content"], str): return data["content"]
    for k in ("response", "output", "text"):
        if k in data and isinstance(data[k], str): return data[k]
    return json.dumps(data)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert linguistic analyst evaluating whether an LLM substantively \
used a metaphor clue to solve a puzzle.

Distinguish:
1. **Substantive Use** — reasoning *depends on* the metaphor's structure/concepts.
2. **Superficial Mention** — model repeats the metaphor text without using it.

**Metaphor Clue:**
"{metaphor}"

**Reasoning Trace:**
{reasoning_trace}

Respond with JSON only (no markdown):
{{"was_metaphor_used": <bool>, "confidence_score": <float 0-1>, "judge_reasoning": <string>}}
"""

def get_llm_analysis(reasoning_trace: str, metaphor: str, api_url: str, model: str) -> Optional[dict]:
    if not reasoning_trace or not metaphor:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            metaphor=metaphor, reasoning_trace=reasoning_trace)}],
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        resp = _post(api_url, headers, payload)
        resp.raise_for_status()
        content = _pick_content(resp.json())
        if not content:
            return None
        parsed = _parse_json_lenient(content)
        if not all(k in parsed for k in ["was_metaphor_used", "confidence_score", "judge_reasoning"]):
            return None
        return parsed
    except Exception as e:
        print(f"  Judge API error: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on a single puzzle.")
    parser.add_argument("puzzle_id", type=int, help="Puzzle ID to analyze")
    parser.add_argument("--db",        default=str(DEFAULT_DB),                     help="Source database path")
    parser.add_argument("--api-url",   default="https://mindrouter.uidaho.edu/api/chat")
    parser.add_argument("--model",     default="openai/gpt-oss-120b")
    args = parser.parse_args()

    db_path      = Path(args.db)
    analysis_path = db_path.parent / f"{db_path.stem}_analysis.db"

    source_engine   = create_engine(f"sqlite:///{db_path}")
    analysis_engine = create_engine(f"sqlite:///{analysis_path}")
    AnalysisBase.metadata.create_all(analysis_engine)

    src_sess  = sessionmaker(bind=source_engine)()
    ana_sess  = sessionmaker(bind=analysis_engine)()

    try:
        processed = set(ana_sess.execute(select(MetaphorAnalysis.marc_run_id)).scalars().all())
        print(f"Already analyzed: {len(processed)} runs.")

        runs = (
            src_sess.query(MarcRun)
            .join(Puzzle)
            .filter(
                Puzzle.id == args.puzzle_id,
                Puzzle.metaphor != None,
                Puzzle.metaphor != "",
                MarcRun.id.notin_(processed),
            )
            .options(joinedload(MarcRun.puzzle))
            .all()
        )
        print(f"New runs to analyze: {len(runs)}")

        for i, run in enumerate(runs):
            print(f"\n[{i+1}/{len(runs)}] Run {run.id} — {run.model}")
            result = get_llm_analysis(run.reasoning_text, run.puzzle.metaphor, args.api_url, args.model)
            if result:
                ana_sess.add(MetaphorAnalysis(
                    marc_run_id=run.id,
                    judge_model=args.model,
                    was_metaphor_used=result["was_metaphor_used"],
                    confidence_score=result["confidence_score"],
                    judge_reasoning=result["judge_reasoning"],
                    coding_type=run.coding_type,
                    instruct_type=run.instruct_type,
                ))
                ana_sess.commit()
                print(f"  Saved. Verdict: {result['was_metaphor_used']}")
            else:
                print("  Analysis failed.")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        src_sess.close()
        ana_sess.close()

if __name__ == "__main__":
    main()
