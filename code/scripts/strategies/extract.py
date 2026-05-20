"""
scripts/strategies/extract.py — LLM Judge: Strategy Extraction
===============================================================
Uses an LLM judge to analyze the reasoning traces of PASSING marc_runs and
extract the algorithmic strategies the model employed. Produces a companion
_STRATEGIES.db. Resumes automatically — already-processed runs are skipped.

Usage:
    python scripts/strategies/extract.py <db_path> [--puzzle-id <id>]
    # e.g. python scripts/strategies/extract.py data/gpt5.db
    # e.g. python scripts/strategies/extract.py data/gpt5.db --puzzle-id 42

Output:
    data/<source>_STRATEGIES.db  — populated with strategy_analysis rows
"""
import json
import re
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
BASE_DIR   = SCRIPT_DIR.parent.parent

CONNECT_SECS   = 10
LONG_READ_SECS = 120

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

class StrategyAnalysis(AnalysisBase):
    __tablename__ = "strategy_analysis"
    id                   = Column(Integer, primary_key=True)
    marc_run_id          = Column(Integer, nullable=False, unique=True)
    puzzle_id            = Column(Integer, nullable=False)
    metaphor             = Column(String, nullable=True)
    instruct_type        = Column(String, nullable=False)
    coding_type          = Column(String, nullable=False)
    judge_model          = Column(String, nullable=False)
    attempted_strategies = Column(Text, nullable=False)
    winning_strategy     = Column(Text, nullable=False)
    analysis_timestamp   = Column(DateTime, default=datetime.now)


def _make_http_session():
    s = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.8,
                  status_forcelist=(502, 503, 504),
                  allowed_methods=frozenset(["POST", "GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

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

JUDGE_PROMPT = """\
You are an expert cognitive scientist analyzing how an LLM solved an ARC-AGI puzzle.

Extract the algorithmic strategies used in the reasoning trace.

1. Identify **all** distinct strategies or hypotheses attempted (including failed ones).
2. Identify the **single winning strategy** that led to the correct solution.

**Reasoning Trace:**
{reasoning_trace}

Respond with JSON only (no markdown):
{{"attempted_strategies": [<list of strings>], "winning_strategy": <string>}}
"""

def get_llm_analysis(reasoning_trace: str, api_url: str, model: str) -> Optional[dict]:
    if not reasoning_trace:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            reasoning_trace=reasoning_trace)}],
        "stream": False,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        resp = _make_http_session().post(
            api_url, headers=headers, json=payload,
            timeout=(CONNECT_SECS, LONG_READ_SECS), stream=False
        )
        resp.raise_for_status()
        content = _pick_content(resp.json())
        if not content:
            return None
        parsed = _parse_json_lenient(content)
        if not all(k in parsed for k in ["attempted_strategies", "winning_strategy"]):
            return None
        return parsed
    except Exception as e:
        print(f"  API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract strategies from successful runs.")
    parser.add_argument("db",          help="Source database path (e.g. data/gpt5.db)")
    parser.add_argument("--puzzle-id", type=int, help="Restrict to one puzzle ID")
    parser.add_argument("--api-url",   default="https://mindrouter.uidaho.edu/api/chat")
    parser.add_argument("--model",     default="openai/gpt-oss-120b")
    args = parser.parse_args()

    db_path       = Path(args.db)
    analysis_path = db_path.parent / f"{db_path.stem}_STRATEGIES.db"

    print(f"Source DB:   {db_path}")
    print(f"Strategy DB: {analysis_path}")

    if not db_path.exists():
        print(f"Error: {db_path} not found.")
        return

    source_engine   = create_engine(f"sqlite:///{db_path}")
    analysis_engine = create_engine(f"sqlite:///{analysis_path}")
    AnalysisBase.metadata.create_all(analysis_engine)

    src_sess = sessionmaker(bind=source_engine)()
    ana_sess = sessionmaker(bind=analysis_engine)()

    try:
        processed = set(ana_sess.execute(select(StrategyAnalysis.marc_run_id)).scalars().all())
        print(f"Already processed: {len(processed)} runs.")

        query = (
            src_sess.query(MarcRun)
            .options(joinedload(MarcRun.puzzle))
            .filter(MarcRun.did_pass == True, MarcRun.id.notin_(processed))
        )
        if args.puzzle_id:
            query = query.filter(MarcRun.puzzle_id == args.puzzle_id)

        runs = query.all()
        print(f"New runs to analyze: {len(runs)}")

        for i, run in enumerate(runs):
            print(f"\n[{i+1}/{len(runs)}] Run {run.id} (Puzzle {run.puzzle_id})")
            result = get_llm_analysis(run.reasoning_text, args.api_url, args.model)
            if result:
                ana_sess.add(StrategyAnalysis(
                    marc_run_id=run.id,
                    puzzle_id=run.puzzle_id,
                    metaphor=run.puzzle.metaphor,
                    instruct_type=run.instruct_type,
                    coding_type=run.coding_type,
                    judge_model=args.model,
                    attempted_strategies=json.dumps(result["attempted_strategies"]),
                    winning_strategy=result["winning_strategy"],
                ))
                ana_sess.commit()
                print(f"  Saved: {result['winning_strategy'][:60]}...")
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
