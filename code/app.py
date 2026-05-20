#!/usr/bin/env python3
"""
app.py — ARC Research Platform (Master Web App)

Run with:
    cd code/BAFL
    streamlit run app.py
"""
import base64
import csv
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from matplotlib import colors
from requests.adapters import HTTPAdapter
from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer,
    String, Text, create_engine, select,
)
from sqlalchemy.orm import (
    declarative_base, joinedload, relationship, selectinload, sessionmaker,
)
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR                     # project root (app.py is at root)
DATA_DIR   = BASE_DIR / "data"

from lib.config import (
    DEFAULT_CONFIG, DEFAULT_PROVIDER_PROFILES,
    PROVIDER_URLS, PROVIDER_ICONS, PROVIDER_COLORS, PROVIDER_DISPLAY_ORDER,
    load_config, save_config,
)
from lib.database import get_session
from lib.visualizer import grid_to_base64
from lib.models import Run
from lib.experiment import EXPERIMENTS, build_prompt, run_single_experiment, save_run

PUZZLES_DIR = BASE_DIR / "puzzles"
EXP_ORDER   = ["Base_Raw", "Base_Metaphor", "CMT_Raw", "CMT_Metaphor"]

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#ADD8E6", "#870C25",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ARC Research Platform", page_icon="🧩", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def get_cfg() -> dict:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load_config()
    return st.session_state["cfg"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🧩 ARC Research")
page = st.sidebar.radio(
    "Navigate",
    ["🧪 BAFL", "🔍 Judge", "📈 Marcability", "🧠 Strategies", "🖼 VLLM", "⚙️ Settings"],
    label_visibility="collapsed",
)

cfg = get_cfg()
st.sidebar.divider()

# ── Quick provider switcher ──────────────────────────────────────────────────
_cur_key = cfg.get("provider", "mindrouter")
_cur_display = _cur_key.title()
if _cur_display not in PROVIDER_DISPLAY_ORDER:
    _cur_display = "Custom"

_new_display = st.sidebar.selectbox(
    "Active Provider",
    PROVIDER_DISPLAY_ORDER,
    index=PROVIDER_DISPLAY_ORDER.index(_cur_display),
    key="sidebar_provider_switcher",
)
_new_key = _new_display.lower()

if _new_key != _cur_key:
    _profiles = cfg.get("provider_profiles", {})
    _profile  = _profiles.get(_new_key, {})
    _new_cfg  = dict(cfg)
    _new_cfg["provider"]   = _new_key
    _new_cfg["api_url"]    = PROVIDER_URLS.get(_new_key, cfg["api_url"])
    _new_cfg["api_key"]    = _profile.get("api_key")    or cfg["api_key"]
    _new_cfg["model_name"] = _profile.get("model_name") or cfg["model_name"]
    save_config(_new_cfg)
    st.session_state["cfg"] = _new_cfg
    cfg = _new_cfg
    st.rerun()

_icon  = PROVIDER_ICONS.get(_new_key, "⚫")
_color = PROVIDER_COLORS.get(_new_key, "#374151")
st.sidebar.markdown(
    f'<div style="background:{_color};color:white;padding:5px 10px;'
    f'border-radius:5px;font-size:0.85rem;margin-top:4px;">'
    f'{_icon} <b>{_new_display}</b><br>'
    f'<span style="opacity:0.85;font-size:0.8rem;">{cfg["model_name"]}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ORM Models — source research DBs
# ---------------------------------------------------------------------------

SourceBase = declarative_base()

class SrcPuzzle(SourceBase):
    __tablename__ = "puzzles"
    id         = Column(Integer, primary_key=True)
    discipline = Column(String, nullable=False)
    arc_type   = Column(Enum("ARC", "ARC2", "MARC", name="arc_types"), nullable=False)
    metaphor   = Column(String, nullable=True)
    file_path  = Column(String, nullable=False)
    runs       = relationship("SrcRun", back_populates="puzzle", cascade="all, delete-orphan")

class SrcRun(SourceBase):
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
    puzzle              = relationship("SrcPuzzle", back_populates="runs")

JudgeBase = declarative_base()

class MetaphorAnalysis(JudgeBase):
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

StratBase = declarative_base()

class StrategyAnalysis(StratBase):
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

VLLMBase = declarative_base()
VLLM_DB_PATH = DATA_DIR / "vllm_runs.db"

class VLLMRun(VLLMBase):
    __tablename__ = "vllm_runs"
    id           = Column(Integer, primary_key=True)
    puzzle_name  = Column(String, nullable=True)
    model        = Column(String, nullable=False)
    api_url      = Column(String, nullable=True)
    reasoning    = Column(Text, nullable=True)
    output_grid  = Column(Text, nullable=True)   # JSON string
    raw_response = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.now)

def _get_vllm_session():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{VLLM_DB_PATH}")
    VLLMBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()

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

def _post_llm(url: str, payload: dict, connect_secs=10, read_secs=120) -> requests.Response:
    sess = _make_http_session()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    return sess.post(url, headers=headers, json=payload,
                     timeout=(connect_secs, read_secs), stream=False)

def _parse_json_lenient(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", s, flags=re.DOTALL)
        return json.loads(s)

def _pick_content(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return str(data)
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        if "content" in msg:
            return msg["content"]
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content")
    if "content" in data and isinstance(data["content"], str):
        return data["content"]
    for k in ("response", "output", "text"):
        if k in data and isinstance(data[k], str):
            return data[k]
    return json.dumps(data)

def _src_session(db_path: Path):
    return sessionmaker(bind=create_engine(f"sqlite:///{db_path}"))()

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _plot_grid(ax, grid, title: str = ""):
    if not grid:
        ax.axis("off"); return
    try:
        arr = np.array(grid, dtype=int)
    except Exception:
        ax.axis("off"); return
    h, w = arr.shape
    ax.imshow(arr, cmap=colors.ListedColormap(ARC_COLORS), norm=colors.Normalize(vmin=0, vmax=9))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, pad=5)

def visualize_puzzle_to_bytes(json_data: dict, show_solution: bool = False) -> Optional[bytes]:
    metaphor    = json_data.get("metaphor", "")
    train_pairs = json_data.get("train", [])
    test_pairs  = json_data.get("test", [])
    num_rows    = len(train_pairs) + len(test_pairs)
    if num_rows == 0:
        return None
    fig = plt.figure(figsize=(10, 3 * num_rows + 1), constrained_layout=True)
    if metaphor:
        fig.suptitle(f"Metaphor: {metaphor}", fontsize=14, wrap=True, fontweight="bold")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    subfigs = fig.subfigures(num_rows, 1, wspace=0.1, hspace=0.1)
    if num_rows == 1:
        subfigs = [subfigs]
    all_pairs = ([(p, "Train", i+1) for i, p in enumerate(train_pairs)]
                 + [(p, "Test",  i+1) for i, p in enumerate(test_pairs)])
    for subfig, (pair, label, count) in zip(subfigs, all_pairs):
        axs = subfig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 0.2, 1]})
        _plot_grid(axs[0], pair.get("input"),  title=f"{label} {count} Input")
        axs[1].axis("off")
        axs[1].text(0.5, 0.5, "→", fontsize=30, ha="center", va="center")
        output = pair.get("output")
        if output and (label == "Train" or show_solution):
            _plot_grid(axs[2], output, title=f"{label} {count} Output")
        else:
            axs[2].axis("off")
            axs[2].text(0.5, 0.5, "?", fontsize=30, ha="center", va="center")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------------------
# Judge logic
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

def run_judge(reasoning_trace: str, metaphor: str, api_url: str, model: str) -> Optional[dict]:
    if not reasoning_trace or not metaphor:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            metaphor=metaphor, reasoning_trace=reasoning_trace)}],
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = _post_llm(api_url, payload)
        resp.raise_for_status()
        content = _pick_content(resp.json())
        if not content:
            return None
        parsed = _parse_json_lenient(content)
        if not {"was_metaphor_used", "confidence_score", "judge_reasoning"}.issubset(parsed):
            return None
        return parsed
    except Exception as e:
        st.warning(f"Judge API error: {e}")
        return None

# ---------------------------------------------------------------------------
# Strategies logic
# ---------------------------------------------------------------------------

STRATEGY_PROMPT = """\
You are an expert cognitive scientist analyzing how an LLM solved an ARC-AGI puzzle.

**Reasoning Trace:**
{reasoning_trace}

Respond with JSON only (no markdown):
{{"attempted_strategies": [<string>, ...], "winning_strategy": <string>}}
"""

def run_strategy_extraction(reasoning_trace: str, api_url: str, model: str) -> Optional[dict]:
    if not reasoning_trace:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": STRATEGY_PROMPT.format(reasoning_trace=reasoning_trace)}],
        "stream": False,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    try:
        resp = _post_llm(api_url, payload, read_secs=180)
        resp.raise_for_status()
        content = _pick_content(resp.json())
        if not content:
            return None
        parsed = _parse_json_lenient(content)
        if not {"attempted_strategies", "winning_strategy"}.issubset(parsed):
            return None
        return parsed
    except Exception as e:
        st.warning(f"Strategy API error: {e}")
        return None

# ---------------------------------------------------------------------------
# Marcability logic
# ---------------------------------------------------------------------------

def compute_marcability(db_path: Path) -> list[dict]:
    session = sessionmaker(bind=create_engine(f"sqlite:///{db_path}"))()
    try:
        puzzles = session.execute(
            select(SrcPuzzle).options(selectinload(SrcPuzzle.runs))
        ).scalars().all()

        def rate(runs): return sum(r.did_pass for r in runs) / len(runs) if runs else None
        def breakdown(runs):
            d: dict = {}
            for r in runs: d.setdefault(r.coding_type, []).append(r)
            return {ct: f"{sum(x.did_pass for x in rs)}/{len(rs)}" for ct, rs in d.items()}

        rows = []
        for p in puzzles:
            meta = [r for r in p.runs if r.instruct_type == "metaphor"]
            none = [r for r in p.runs if r.instruct_type == "none"]
            rows.append({
                "puzzle_id": p.id, "metaphor": p.metaphor or "",
                "metaphor_rate": rate(meta), "none_rate": rate(none),
                "metaphor_count": len(meta), "none_count": len(none),
                "metaphor_breakdown": breakdown(meta), "none_breakdown": breakdown(none),
            })
        return rows
    finally:
        session.close()

# ---------------------------------------------------------------------------
# VLLM logic
# ---------------------------------------------------------------------------

VLLM_PROMPT = """\
This is an ARC-style puzzle. Using Conceptual Metaphor Theory (CMT), interpret \
abstract concepts through concrete experiences.

Analyze the image. It contains training examples and a test input.
Find the metaphor box in the image and use it to guide your reasoning.

Output ONLY a JSON object:
{"reasoning": "<your reasoning>", "output_grid": [[<int>, ...], ...]}
"""

def run_vllm(image_bytes: bytes, api_url: str, model: str, prompt: str = VLLM_PROMPT) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False, "options": {"temperature": 0.1},
    }
    try:
        resp = _post_llm(api_url, payload, read_secs=1200)
        resp.raise_for_status()
        data = resp.json()
        content = thinking = ""
        if "message" in data:
            content  = data["message"].get("content", "")
            thinking = data["message"].get("thinking", "")
        full = (thinking + "\n" + content).strip() if thinking else content
        clean = full.strip()
        if clean.startswith("```"): clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):   clean = clean.rsplit("\n", 1)[0]
        try:
            parsed = json.loads(clean)
            return {"reasoning": parsed.get("reasoning", full),
                    "output_grid": parsed.get("output_grid", []),
                    "raw": full, "error": None}
        except json.JSONDecodeError:
            return {"reasoning": full, "output_grid": [], "raw": full, "error": None}
    except Exception as e:
        return {"reasoning": "", "output_grid": [], "raw": "", "error": str(e)}

# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def _grid_image(grid_json_str: str):
    if not grid_json_str or grid_json_str == "[]":
        return
    try:
        grid = json.loads(grid_json_str)
        if grid:
            b64 = grid_to_base64(grid)
            if b64:
                st.image(base64.b64decode(b64), use_container_width=False)
    except Exception:
        pass

def _load_puzzle_data(puzzle_id: str):
    path = PUZZLES_DIR / puzzle_id
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _source_db_selector(key: str, label: str = "Source database") -> Optional[Path]:
    if DATA_DIR.exists():
        db_files = sorted(p for p in DATA_DIR.glob("*.db")
                          if not p.stem.endswith("_analysis")
                          and not p.stem.endswith("_STRATEGIES")
                          and p.name not in ("bafl.db", "vllm_runs.db"))
    else:
        db_files = []
    if db_files:
        names = [p.name for p in db_files]
        sel = st.selectbox(label, names, key=f"{key}_sel")
        return DATA_DIR / sel
    else:
        val = st.text_input(label, value=str(DATA_DIR / "gpt5.db"), key=f"{key}_txt")
        p = Path(val.strip())
        return p if p.exists() else None

def _export_html_button(label: str = "📄 Export HTML Report"):
    if st.button(label, type="secondary"):
        with st.spinner("Generating report…"):
            try:
                import zipfile, tempfile
                from lib import report as make_report
                with tempfile.TemporaryDirectory() as tmp:
                    make_report.main(["--output-dir", tmp])
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for f in Path(tmp).glob("*.html"):
                            zf.write(f, f.name)
                    buf.seek(0)
                st.download_button("⬇️ Download report.zip", buf, "bafl_report.zip",
                                   "application/zip", key=f"dl_{label}")
            except Exception as e:
                st.error(f"Export failed: {e}")

# ---------------------------------------------------------------------------
# BAFL sub-tabs helpers
# ---------------------------------------------------------------------------

def _provider_banner(cfg: dict) -> None:
    """Colored banner showing active provider, model, and endpoint."""
    provider = cfg.get("provider", "mindrouter")
    icon     = PROVIDER_ICONS.get(provider, "⚫")
    color    = PROVIDER_COLORS.get(provider, "#374151")
    model    = cfg.get("model_name", "unknown")
    url      = cfg.get("api_url", "")
    st.markdown(
        f'<div style="background:{color};color:white;padding:10px 14px;'
        f'border-radius:6px;margin-bottom:12px;font-size:0.9rem;">'
        f'{icon} <b>{provider.title()}</b>&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<code style="background:rgba(255,255,255,0.2);padding:2px 6px;border-radius:3px;">{model}</code>'
        f'<span style="opacity:0.75;font-size:0.78rem;margin-left:12px;">{url}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _run_tab():
    _provider_banner(cfg)
    col_setup, col_output = st.columns([1, 2], gap="large")
    with col_setup:
        st.subheader("Puzzle Input")
        input_mode = st.radio("Input mode", ["Single File", "Directory"], horizontal=True)
        puzzle_paths = []
        if input_mode == "Single File":
            uploaded   = st.file_uploader("Upload puzzle JSON", type=["json"])
            manual_path = st.text_input("…or enter a file path")
            if uploaded:
                try:
                    st.session_state["_uploaded_puzzle"] = {
                        "name": uploaded.name, "data": json.loads(uploaded.read())}
                    puzzle_paths = [f"__uploaded__:{uploaded.name}"]
                except Exception as e:
                    st.error(f"Could not parse uploaded file: {e}")
            elif manual_path.strip():
                puzzle_paths = [manual_path.strip()]
        else:
            dir_path = st.text_input("Directory path", value=str(PUZZLES_DIR))
            if dir_path.strip() and os.path.isdir(dir_path.strip()):
                puzzle_paths = sorted(str(p) for p in Path(dir_path.strip()).glob("*.json"))
                (st.success if puzzle_paths else st.warning)(
                    f"Found {len(puzzle_paths)} puzzle(s)" if puzzle_paths else "No .json files found")
            elif dir_path.strip():
                st.error("Directory not found")

        st.subheader("Experiments")
        exp_checks   = {e["name"]: st.checkbox(e["name"].replace("_", " "), value=True) for e in EXPERIMENTS}
        selected_exps = [e for e in EXPERIMENTS if exp_checks[e["name"]]]
        st.divider()
        run_clicked = st.button("▶ Run", type="primary",
                                disabled=not puzzle_paths or not selected_exps)

    with col_output:
        st.subheader("Live Output")
        if not run_clicked:
            st.info("Configure a puzzle and click Run.")
            return

        total_steps = len(puzzle_paths) * len(selected_exps)
        progress    = st.progress(0)
        status_text = st.empty()
        step = 0

        for puzzle_path in puzzle_paths:
            if puzzle_path.startswith("__uploaded__:"):
                up = st.session_state.get("_uploaded_puzzle", {})
                puzzle_id, puzzle_data = up.get("name", "uploaded.json"), up.get("data")
                if not puzzle_data:
                    st.error("Uploaded puzzle data missing — please re-upload.")
                    step += len(selected_exps); progress.progress(step / total_steps); continue
            else:
                puzzle_id = os.path.basename(puzzle_path)
                try:
                    puzzle_data = json.loads(Path(puzzle_path).read_text(encoding="utf-8"))
                except Exception as e:
                    st.error(f"Could not load {puzzle_id}: {e}")
                    step += len(selected_exps); progress.progress(step / total_steps); continue

            st.markdown(f"### 🧩 `{puzzle_id}`")
            try:
                correct_grid = puzzle_data["test"][0]["output"]
            except (KeyError, IndexError):
                st.error(f"{puzzle_id}: missing test output")
                step += len(selected_exps); progress.progress(step / total_steps); continue

            run_results = []
            for exp in selected_exps:
                status_text.text(f"Running {exp['name']} on {puzzle_id}…")
                result = run_single_experiment(puzzle_data, exp, correct_grid, cfg)
                if result["status_str"] == "PASS":
                    st.success(f"✅ {exp['name']} — PASS")
                elif result["status_str"] == "FAIL":
                    st.error(f"❌ {exp['name']} — FAIL")
                else:
                    st.warning(f"⚠️ {exp['name']} — ERROR: {result['error']}")
                if result["status_str"] != "ERR":
                    try:
                        save_run(result, puzzle_id, exp, cfg)
                    except Exception as e:
                        st.warning(f"DB save failed: {e}")
                run_results.append(result)
                step += 1; progress.progress(step / total_steps)

            with st.expander("View grid comparison"):
                grid_cols = st.columns(len(run_results) + 1)
                with grid_cols[0]:
                    st.caption("**Correct Answer**")
                    b64 = grid_to_base64(correct_grid)
                    if b64: st.image(base64.b64decode(b64), use_container_width=True)
                for i, result in enumerate(run_results):
                    with grid_cols[i + 1]:
                        badge = "✅" if result["pass_fail"] else "❌"
                        st.caption(f"**{badge} {result['experiment_type'].replace('_', ' ')}**")
                        if result["ai_grid"]:
                            b64 = grid_to_base64(result["ai_grid"])
                            if b64: st.image(base64.b64decode(b64), use_container_width=True)
                        else:
                            st.caption("No grid")

        status_text.text("Done!"); progress.progress(1.0)
        st.success("All experiments complete.")

def _results_tab():
    h_col, export_col = st.columns([4, 1])
    with h_col: st.subheader("Experiment Results")
    with export_col: st.write(""); _export_html_button()

    session  = get_session()
    all_runs = [r.to_dict() for r in session.query(Run).all()]
    session.close()

    if not all_runs:
        st.info("No results yet. Run some experiments first.")
        return

    all_models  = sorted({r["model"]     for r in all_runs})
    all_puzzles = sorted({r["puzzle_id"] for r in all_runs})

    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1: model_filter  = st.multiselect("Model",  all_models,  default=all_models,  key="res_m")
        with fc2: puzzle_filter = st.multiselect("Puzzle", all_puzzles, default=all_puzzles, key="res_p")
        with fc3: exp_filter    = st.multiselect("Experiment Type", EXP_ORDER, default=EXP_ORDER, key="res_e")

    rows = [r for r in all_runs
            if r["model"] in model_filter
            and r["puzzle_id"] in puzzle_filter
            and r["experiment_type"] in exp_filter]

    if not rows:
        st.warning("No runs match the current filters.")
        return

    total  = len(rows); passes = sum(1 for r in rows if r["pass_fail"] == 1)
    unique_puzzles = {r["puzzle_id"] for r in rows}

    # Pass@1: per (puzzle, model, experiment_type) group, did any run pass?
    p1_groups: dict = {}
    for r in rows:
        key = (r["puzzle_id"], r["model"], r["experiment_type"])
        p1_groups[key] = p1_groups.get(key, False) or bool(r["pass_fail"])
    p1_total  = len(p1_groups)
    p1_passes = sum(p1_groups.values())

    view_mode = st.radio("View", ["Pass Rate (consistency)", "Pass@1 (ARC-AGI)"],
                         horizontal=True, key="res_view")
    using_p1 = view_mode == "Pass@1 (ARC-AGI)"

    if using_p1:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Puzzle×Model×Exp Groups", p1_total)
        s2.metric("Groups with ≥1 Pass", p1_passes)
        s3.metric("Pass@1", f"{p1_passes/p1_total*100:.1f}%" if p1_total else "—")
        s4.metric("Unique Puzzles", len(unique_puzzles))
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Runs", total); s2.metric("Passes", passes)
        s3.metric("Accuracy",   f"{passes/total*100:.1f}%")
        s4.metric("Unique Puzzles", len(unique_puzzles))

    chart_label = "Pass@1 (%)" if using_p1 else "Pass Rate (%)"
    st.subheader(f"{chart_label.split(' (')[0]} by Experiment Type")
    chart_data = []
    for model in model_filter:
        for exp_name in EXP_ORDER:
            if exp_name not in exp_filter: continue
            subset = [r for r in rows if r["model"] == model and r["experiment_type"] == exp_name]
            if not subset: continue
            if using_p1:
                puzzles_in_subset = {r["puzzle_id"] for r in subset}
                p1_passed = sum(
                    any(r["pass_fail"] for r in subset if r["puzzle_id"] == pid)
                    for pid in puzzles_in_subset
                )
                rate = round(p1_passed / len(puzzles_in_subset) * 100, 1)
                n_label = len(puzzles_in_subset)
            else:
                rate = round(sum(r["pass_fail"] for r in subset) / len(subset) * 100, 1)
                n_label = len(subset)
            chart_data.append({
                "Experiment": exp_name.replace("_", " "),
                chart_label: rate,
                "Model": model, "n": n_label,
            })
    if chart_data:
        fig = px.bar(chart_data, x="Experiment", y=chart_label,
                     color="Model" if len(model_filter) > 1 else None,
                     barmode="group", text=chart_label, range_y=[0, 100], hover_data=["n"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Pass@1 per Puzzle per Model ──────────────────────────────────────────
    if using_p1:
        st.subheader("Pass@1 by Puzzle — per Model")
        puzzle_ids  = sorted({r["puzzle_id"] for r in rows})
        active_exps = [e for e in EXP_ORDER if e in exp_filter]
        exp_colors  = {"Base_Raw": "#4e79a7", "Base_Metaphor": "#76b7b2",
                       "CMT_Raw":  "#f28e2b", "CMT_Metaphor":  "#59a14f"}

        # Assign a numeric index to each puzzle — shared across all charts
        puzzle_index = {pid: i + 1 for i, pid in enumerate(puzzle_ids)}

        # ── Puzzle key as a rendered table ───────────────────────────────────
        n_cols   = 3
        n_rows   = (len(puzzle_ids) + n_cols - 1) // n_cols
        key_data = [[""] * (n_cols * 2) for _ in range(n_rows)]
        for i, pid in enumerate(puzzle_ids):
            col_pair = (i % n_cols) * 2
            row      = i // n_cols
            key_data[row][col_pair]     = str(puzzle_index[pid])
            key_data[row][col_pair + 1] = pid

        fig_key, ax_key = plt.subplots(figsize=(11, max(1.0, n_rows * 0.38)))
        fig_key.patch.set_facecolor("#0e1117")
        ax_key.set_facecolor("#0e1117")
        ax_key.axis("off")

        tbl = ax_key.table(cellText=key_data, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for (row, col), cell in tbl.get_celld().items():
            is_idx = (col % 2 == 0)
            cell.set_facecolor("#161b22")
            cell.set_edgecolor("#30363d")
            cell.set_text_props(
                color="#8b949e" if is_idx else "#e6edf3",
                fontweight="bold" if is_idx else "normal",
            )
            if is_idx:
                cell.set_width(0.04)
        ax_key.set_title("Puzzle Key", color="#e6edf3", fontsize=10, pad=4)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_key, use_container_width=True)
        plt.close(fig_key)

        st.divider()

        # Four clearly distinct, bright colors for the experiment types
        exp_colors = {
            "Base_Raw":      "#4C9BE8",   # blue
            "Base_Metaphor": "#B07FE8",   # purple
            "CMT_Raw":       "#F0A500",   # amber
            "CMT_Metaphor":  "#3EC97A",   # green
        }

        for model in model_filter:
            data = {}
            for puzzle_id in puzzle_ids:
                data[puzzle_id] = {}
                for exp_name in active_exps:
                    exp_runs = [r for r in rows
                                if r["puzzle_id"] == puzzle_id
                                and r["model"] == model
                                and r["experiment_type"] == exp_name]
                    if exp_runs:
                        data[puzzle_id][exp_name] = 1 if any(r["pass_fail"] for r in exp_runs) else 0

            puzzles_with_data = [p for p in puzzle_ids if data[p]]
            if not puzzles_with_data:
                st.caption(f"**{model}** — no data"); continue

            n_puzzles = len(puzzles_with_data)
            n_exps    = len(active_exps)
            bar_w     = 0.7 / n_exps
            xs        = list(range(n_puzzles))

            fig, ax = plt.subplots(figsize=(max(6, n_puzzles * 0.6), 3.2))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            for ei, exp_name in enumerate(active_exps):
                color   = exp_colors.get(exp_name, "#aaaaaa")
                offsets = [x + (ei - (n_exps - 1) / 2) * bar_w for x in xs]
                vals    = [data[p].get(exp_name, None) for p in puzzles_with_data]

                for x, v in zip(offsets, vals):
                    if v == 1:
                        # Pass: full bright bar
                        ax.bar(x, 1.0, width=bar_w * 0.85, color=color,
                               label=exp_name.replace("_", " ") if x == offsets[0] else "")
                    elif v == 0:
                        # Fail: faint outline bar only
                        ax.bar(x, 1.0, width=bar_w * 0.85, color="none",
                               edgecolor=color, linewidth=1.2, linestyle="--",
                               label=exp_name.replace("_", " ") if x == offsets[0] else "")
                    else:
                        # Not attempted: small gray stub
                        ax.bar(x, 0.08, width=bar_w * 0.85, color="#2a2a2a")

            ax.set_xticks(xs)
            ax.set_xticklabels([str(puzzle_index[p]) for p in puzzles_with_data],
                               fontsize=10, color="#cccccc")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Fail", "Pass"], color="#cccccc", fontsize=9)
            ax.set_ylim(0, 1.35)
            ax.set_title(model, color="#eeeeee", fontsize=11, pad=22)
            ax.tick_params(colors="#444444", length=0)
            for spine in ax.spines.values(): spine.set_edgecolor("#2a2a2a")
            ax.yaxis.grid(True, color="#1e1e1e", linewidth=0.8, linestyle="--")
            ax.set_axisbelow(True)
            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels):
                if l not in seen: seen[l] = h
            ax.legend(seen.values(), seen.keys(),
                      ncol=n_exps, loc="upper center", bbox_to_anchor=(0.5, 1.18),
                      fontsize=8, facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#cccccc", handlelength=1.5, handleheight=1.0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Puzzle Breakdown")
    grouped: dict = defaultdict(list)
    for r in rows: grouped[r["puzzle_id"]].append(r)

    for puzzle_id, p_rows in sorted(grouped.items()):
        puzzle_passes = sum(r["pass_fail"] for r in p_rows)
        puzzle_models = sorted({r["model"] for r in p_rows})
        if using_p1:
            # count (model, exp_type) groups that passed at least once
            p1_puzzle_groups = {}
            for r in p_rows:
                k = (r["model"], r["experiment_type"])
                p1_puzzle_groups[k] = p1_puzzle_groups.get(k, False) or bool(r["pass_fail"])
            p1_puzzle_pass = sum(p1_puzzle_groups.values())
            p1_puzzle_total = len(p1_puzzle_groups)
            expander_label = f"🧩 {puzzle_id}  —  Pass@1: {p1_puzzle_pass}/{p1_puzzle_total} groups  ({len(puzzle_models)} model{'s' if len(puzzle_models)!=1 else ''})"
        else:
            expander_label = f"🧩 {puzzle_id}  —  {puzzle_passes}/{len(p_rows)} pass  ({len(puzzle_models)} model{'s' if len(puzzle_models)!=1 else ''})"
        with st.expander(expander_label):
            pdata = _load_puzzle_data(puzzle_id)
            if pdata:
                mc1, mc2 = st.columns([2, 3])
                with mc1:
                    if m := pdata.get("metaphor"): st.info(f"**Metaphor:** {m}")
                with mc2:
                    cg = pdata.get("test", [{}])[0].get("output")
                    if cg:
                        st.caption("**Correct Answer**")
                        b64 = grid_to_base64(cg)
                        if b64: st.image(base64.b64decode(b64), width=200)

            visible_exps = [e for e in EXP_ORDER if e in exp_filter]
            for tab, model in zip(st.tabs(puzzle_models), puzzle_models):
                with tab:
                    model_rows = [r for r in p_rows if r["model"] == model]
                    if using_p1:
                        p1_model = {e: any(r["pass_fail"] for r in model_rows if r["experiment_type"] == e)
                                    for e in visible_exps if any(r["experiment_type"] == e for r in model_rows)}
                        st.caption(f"Pass@1: {sum(p1_model.values())}/{len(p1_model)} experiment types")
                    else:
                        st.caption(f"{sum(r['pass_fail'] for r in model_rows)}/{len(model_rows)} pass")
                    exp_cols = st.columns(len(visible_exps))
                    for i, exp_name in enumerate(visible_exps):
                        exp_runs = [r for r in model_rows if r["experiment_type"] == exp_name]
                        with exp_cols[i]:
                            st.markdown(f"**{exp_name.replace('_', ' ')}**")
                            if not exp_runs:
                                st.caption("No data"); continue
                            for run in exp_runs:
                                run_date = (run.get("created_at") or "")[:16]
                                (st.success if run["pass_fail"] else st.error)("PASS" if run["pass_fail"] else "FAIL")
                                if run_date: st.caption(run_date)
                                if run.get("read_timeout"): st.caption(f"⏱ {run['read_timeout']}s timeout")
                                _grid_image(run.get("output_grid"))
                                with st.popover("Reasoning"):
                                    st.text(run.get("reasoning_trace") or "None")
                                if pdata:
                                    with st.popover("Prompt"):
                                        try:
                                            sp, up = build_prompt(pdata, "CMT" in exp_name, "Metaphor" in exp_name, cfg)
                                            st.text(f"--- SYSTEM ---\n{sp}\n\n--- USER ---\n{up}")
                                        except Exception as e:
                                            st.text(f"Error: {e}")
                                if st.button("🗑", key=f"del_{run['id']}", help="Delete"):
                                    s = get_session()
                                    s.query(Run).filter(Run.id == run["id"]).delete()
                                    s.commit(); s.close(); st.rerun()
                                st.divider()

def _manager_tab():
    session  = get_session()
    all_runs = [r.to_dict() for r in session.query(Run).order_by(Run.created_at.desc()).all()]
    session.close()

    if not all_runs:
        st.info("No runs in the database yet.")
        return

    cs, ce = st.columns([3, 1])
    with cs:
        st.caption(f"**{len(all_runs)} total runs** across "
                   f"{len({r['model'] for r in all_runs})} model(s) and "
                   f"{len({r['puzzle_id'] for r in all_runs})} puzzle(s)")
    with ce: _export_html_button("📄 Export HTML")

    # ── Coverage Matrix ──────────────────────────────────────────────────────
    st.subheader("Coverage Matrix")
    target = st.number_input("Target runs per experiment", min_value=1, max_value=50,
                             value=1, step=1, key="rm_target",
                             help="Green = at or above target, yellow = partial, red = missing")

    _exp_cols = EXP_ORDER  # ["Base_Raw", "Base_Metaphor", "CMT_Raw", "CMT_Metaphor"]
    _exp_display = [e.replace("_", " ") for e in _exp_cols]

    _counts: dict = {}
    for r in all_runs:
        key = (r["puzzle_id"], r["model"])
        _counts.setdefault(key, {e: 0 for e in _exp_cols})
        if r["experiment_type"] in _counts[key]:
            _counts[key][r["experiment_type"]] += 1

    _matrix_rows = []
    for (puzzle, model), exp_counts in sorted(_counts.items()):
        row = {"Puzzle": puzzle, "Model": model}
        row.update({e.replace("_", " "): exp_counts[e] for e in _exp_cols if e in exp_counts})
        row["Total"] = sum(exp_counts.values())
        _matrix_rows.append(row)

    _df = pd.DataFrame(_matrix_rows)

    def _cell_color(val):
        if not isinstance(val, (int, float)):
            return ""
        if val == 0:
            return "background-color:#fee2e2;color:#991b1b"
        if val < target:
            return "background-color:#fef9c3;color:#854d0e"
        return "background-color:#dcfce7;color:#166534"

    _styled = _df.style.map(_cell_color, subset=_exp_display + ["Total"])
    st.dataframe(_styled, use_container_width=True, hide_index=True)

    st.divider()
    all_models  = sorted({r["model"]          for r in all_runs})
    all_puzzles = sorted({r["puzzle_id"]       for r in all_runs})
    all_exps    = sorted({r["experiment_type"] for r in all_runs})

    fc1, fc2, fc3 = st.columns(3)
    with fc1: f_models  = st.multiselect("Model",           all_models,  default=all_models,  key="rm_m")
    with fc2: f_puzzles = st.multiselect("Puzzle",          all_puzzles, default=all_puzzles, key="rm_p")
    with fc3: f_exps    = st.multiselect("Experiment Type", all_exps,    default=all_exps,    key="rm_e")
    f_pass = st.radio("Result", ["All", "Pass only", "Fail only"], horizontal=True, key="rm_r")

    rows = [r for r in all_runs
            if r["model"] in f_models and r["puzzle_id"] in f_puzzles and r["experiment_type"] in f_exps
            and (f_pass == "All"
                 or (f_pass == "Pass only" and r["pass_fail"] == 1)
                 or (f_pass == "Fail only" and r["pass_fail"] == 0))]

    st.caption(f"Showing {len(rows)} of {len(all_runs)} runs")
    if st.button(f"🗑 Delete all {len(rows)} shown runs", type="secondary", disabled=not rows):
        ids = [r["id"] for r in rows]
        s   = get_session()
        s.query(Run).filter(Run.id.in_(ids)).delete(synchronize_session=False)
        s.commit(); s.close(); st.success(f"Deleted {len(ids)} runs."); st.rerun()

    st.divider()
    for run in rows:
        icon     = "✅" if run["pass_fail"] == 1 else "❌"
        date_str = (run.get("created_at") or "")[:16]
        with st.expander(f"{icon}  {run['puzzle_id']}  ·  {run['experiment_type'].replace('_',' ')}  ·  {run['model']}  ·  {date_str}", expanded=False):
            left, right = st.columns([1, 2])
            with left:
                st.markdown(f"**Result:** {'PASS ✅' if run['pass_fail'] else 'FAIL ❌'}")
                st.markdown(f"**Model:** `{run['model']}`")
                st.markdown(f"**Experiment:** {run['experiment_type'].replace('_', ' ')}")
                timeout_str = f"{run['read_timeout']}s" if run.get('read_timeout') else "—"
                st.markdown(f"**Puzzle:** `{run['puzzle_id']}`  |  **ID:** {run['id']}  |  **Timeout:** {timeout_str}  |  {date_str}")
                if st.button("🗑 Delete", key=f"rm_del_{run['id']}"):
                    s = get_session()
                    s.query(Run).filter(Run.id == run["id"]).delete()
                    s.commit(); s.close(); st.rerun()
            with right:
                gc, cc = st.columns(2)
                with gc:
                    st.caption("**Model Output**"); _grid_image(run.get("output_grid"))
                with cc:
                    pd_ = _load_puzzle_data(run["puzzle_id"])
                    if pd_:
                        cg = pd_.get("test", [{}])[0].get("output")
                        if cg:
                            st.caption("**Correct Answer**")
                            b64 = grid_to_base64(cg)
                            if b64: st.image(base64.b64decode(b64), use_container_width=True)
            with st.expander("Reasoning trace"):
                st.text(run.get("reasoning_trace") or "None")
            pd_ = _load_puzzle_data(run["puzzle_id"])
            if pd_:
                with st.expander("Prompt"):
                    try:
                        sp, up = build_prompt(pd_, "CMT" in run["experiment_type"], "Metaphor" in run["experiment_type"], cfg)
                        st.text(f"--- SYSTEM ---\n{sp}\n\n--- USER ---\n{up}")
                    except Exception as e:
                        st.text(f"Error: {e}")

# ---------------------------------------------------------------------------
# Auto-Fill tab
# ---------------------------------------------------------------------------

def _autofill_tab():
    _provider_banner(cfg)
    st.markdown(
        "Pick models, puzzles, and experiments below. "
        "Auto-Fill checks existing run counts and only runs what is needed to reach your target."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Models")
        session = get_session()
        db_models = sorted({r.model for r in session.query(Run).all()})
        session.close()
        current_model = cfg.get("model_name", "")
        models_text = st.text_area(
            "Model names (one per line)",
            value=current_model,
            height=130,
            help="Uses the current provider URL and API key — only the model name changes per row.",
        )
        selected_models = [m.strip() for m in models_text.splitlines() if m.strip()]
        if not selected_models:
            st.warning("Enter at least one model name.")

        st.subheader("Experiments")
        exp_checks = {
            e["name"]: st.checkbox(e["name"].replace("_", " "), value=True, key=f"af_exp_{e['name']}")
            for e in EXPERIMENTS
        }
        selected_exps = [e for e in EXPERIMENTS if exp_checks[e["name"]]]

        target = st.number_input(
            "Target runs per experiment", min_value=1, max_value=50, value=3, step=1, key="af_target"
        )

    with col_right:
        st.subheader("Puzzles")
        puzzle_files = sorted(p.name for p in PUZZLES_DIR.glob("*.json")) if PUZZLES_DIR.exists() else []
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Select All", key="af_sel_all"):
                st.session_state["af_puzzles"] = puzzle_files
        with bc2:
            if st.button("Deselect All", key="af_desel_all"):
                st.session_state["af_puzzles"] = []
        selected_puzzles = st.multiselect(
            f"Puzzles  ({len(puzzle_files)} available)",
            puzzle_files,
            default=puzzle_files,
            key="af_puzzles",
        )

    st.divider()

    can_run = bool(selected_models and selected_puzzles and selected_exps)

    def _compute_gaps():
        session = get_session()
        runs = session.query(Run).filter(
            Run.puzzle_id.in_(selected_puzzles),
            Run.model.in_(selected_models),
        ).all()
        session.close()
        counts: dict = {}
        for r in runs:
            key = (r.puzzle_id, r.model, r.experiment_type)
            counts[key] = counts.get(key, 0) + 1
        gaps = []
        for model in selected_models:
            for puzzle_id in selected_puzzles:
                for exp in selected_exps:
                    current = counts.get((puzzle_id, model, exp["name"]), 0)
                    for _ in range(max(0, target - current)):
                        gaps.append({"model": model, "puzzle_id": puzzle_id, "exp": exp})
        return gaps

    pc, sc = st.columns(2)
    with pc:
        preview_clicked = st.button("🔍 Preview Gaps", disabled=not can_run)
    with sc:
        start_clicked = st.button("▶ Start Auto-Fill", type="primary", disabled=not can_run)

    if not can_run:
        st.info("Select at least one model, puzzle, and experiment type to continue.")
        return

    if not (preview_clicked or start_clicked):
        return

    gaps = _compute_gaps()

    if not gaps:
        st.success(
            f"✅ All targets already met — every selected (puzzle × model × experiment) "
            f"has at least {target} run(s)."
        )
        return

    total = len(gaps)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Runs needed", total)
    mc2.metric("Models", len({g["model"] for g in gaps}))
    mc3.metric("Puzzles", len({g["puzzle_id"] for g in gaps}))
    mc4.metric("Target per experiment", target)

    # Preview table — runs still needed per (puzzle, model)
    preview_data: dict = {}
    for g in gaps:
        key = (g["puzzle_id"], g["model"])
        preview_data.setdefault(key, {e: 0 for e in EXP_ORDER})
        if g["exp"]["name"] in preview_data[key]:
            preview_data[key][g["exp"]["name"]] += 1
    preview_rows = []
    for (puzzle, model), ec in sorted(preview_data.items()):
        row = {"Puzzle": puzzle, "Model": model}
        row.update({e.replace("_", " "): ec.get(e, 0) for e in EXP_ORDER})
        row["Total"] = sum(ec.values())
        preview_rows.append(row)
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    if preview_clicked:
        st.info("Click **▶ Start Auto-Fill** to execute these runs.")
        return

    # ── Kick off a new run batch ──────────────────────────────────────────────
    st.session_state["af_gaps"]      = gaps
    st.session_state["af_index"]     = 0
    st.session_state["af_completed"] = 0
    st.session_state["af_errors"]    = []
    st.session_state["af_halted"]    = False
    st.session_state["af_running"]   = True
    st.rerun()

# ── Execute loop (one gap per rerun so Stop fires between API calls) ──────
if st.session_state.get("af_running"):
    gaps      = st.session_state["af_gaps"]
    i         = st.session_state["af_index"]
    total     = len(gaps)
    completed = st.session_state["af_completed"]
    errors    = st.session_state["af_errors"]

    st.subheader("Running…")
    progress    = st.progress(i / total if total else 1.0)
    status_text = st.empty()

    if st.button("🛑 Stop", type="secondary", key="af_stop"):
        st.session_state["af_halted"]  = True
        st.session_state["af_running"] = False
        st.warning(
            f"Stopped after {completed}/{total} runs. "
            "No further API calls will be made."
        )
        if errors:
            for err in errors:
                st.caption(f"• {err}")
        st.stop()

    if st.session_state.get("af_halted"):
        st.session_state["af_running"] = False
        st.stop()

    if i >= total:
        st.session_state["af_running"] = False
        status_text.text("Done!")
        if errors:
            st.warning(f"Completed {completed}/{total} runs with {len(errors)} error(s):")
            for err in errors:
                st.caption(f"• {err}")
        else:
            st.success(f"✅ Auto-Fill complete! {completed} run(s) added.")
        st.stop()

    gap = gaps[i]
    model, puzzle_id, exp = gap["model"], gap["puzzle_id"], gap["exp"]
    status_text.text(f"[{i + 1}/{total}]  {exp['name']}  ·  {puzzle_id}  ·  {model}")

    puzzle_path = PUZZLES_DIR / puzzle_id
    try:
        puzzle_data  = json.loads(puzzle_path.read_text(encoding="utf-8"))
        correct_grid = puzzle_data["test"][0]["output"]
        model_cfg              = dict(cfg)
        model_cfg["model_name"] = model
        result = run_single_experiment(puzzle_data, exp, correct_grid, model_cfg)
        if result["status_str"] == "ERR":
            st.session_state["af_errors"].append(
                f"{puzzle_id} / {model} / {exp['name']}: {result['error']}"
            )
        else:
            save_run(result, puzzle_id, exp, model_cfg)
            st.session_state["af_completed"] += 1
    except Exception as e:
        st.session_state["af_errors"].append(f"{puzzle_id}: {e}")

    st.session_state["af_index"] += 1
    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: BAFL (container for sub-tabs)
# ---------------------------------------------------------------------------

def page_bafl():
    st.header("🧪 BAFL — CMT Experiment Runner")
    tab_run, tab_results, tab_manager, tab_autofill = st.tabs(
        ["▶ Run", "📊 Results", "🗂 Run Manager", "🤖 Auto-Fill"]
    )
    with tab_run:      _run_tab()
    with tab_results:  _results_tab()
    with tab_manager:  _manager_tab()
    with tab_autofill: _autofill_tab()

# ---------------------------------------------------------------------------
# PAGE: Judge
# ---------------------------------------------------------------------------

def _judge_summary_table(data: dict) -> pd.DataFrame:
    """Build summary DataFrame: instruct_type × coding_type stats."""
    rows = []
    for instruct, coding_dict in data.items():
        for coding, d in coding_dict.items():
            t = d["total"]
            rows.append({
                "Instruct Type": instruct, "Coding Type": coding,
                "Total": t,
                "Passed": d["passed"],
                "Pass %": round(d["passed"] / t * 100, 1) if t else 0,
                "Metaphor Used": d["used"],
                "Usage %": round(d["used"] / t * 100, 1) if t else 0,
                "Avg Confidence": round(d["confidence_sum"] / t, 3) if t else 0,
            })
    return pd.DataFrame(rows)

def _contingency_table(ct: dict):
    """Render a 2×2 contingency table using Streamlit columns."""
    total = ct["true_true"] + ct["false_true"] + ct["true_false"] + ct["false_false"]
    def pct(n): return f"{n/total*100:.1f}%" if total else "—"

    st.markdown("**Pass × Metaphor Correlation**")
    header, c1, c2 = st.columns([1.5, 1, 1])
    header.markdown(""); c1.markdown("**Passed ✅**"); c2.markdown("**Failed ❌**")
    r1, v11, v12 = st.columns([1.5, 1, 1])
    r1.markdown("**Metaphor Used ✅**")
    v11.metric("",  ct["true_true"],  help=pct(ct["true_true"]))
    v12.metric("",  ct["true_false"], help=pct(ct["true_false"]))
    r2, v21, v22 = st.columns([1.5, 1, 1])
    r2.markdown("**Metaphor Not Used ❌**")
    v21.metric("", ct["false_true"],  help=pct(ct["false_true"]))
    v22.metric("", ct["false_false"], help=pct(ct["false_false"]))
    st.caption(f"Total: {total}  |  Hover metrics for percentages")

def _build_judge_report_data(a_sess, s_sess):
    """Compile all data structures needed for the Judge report view."""
    analyses = a_sess.execute(select(MetaphorAnalysis)).scalars().all()
    if not analyses:
        return None, None, None

    run_ids   = {a.marc_run_id for a in analyses}
    src_runs  = (s_sess.query(SrcRun)
                 .options(joinedload(SrcRun.puzzle))
                 .filter(SrcRun.id.in_(run_ids))
                 .all())
    run_map = {r.id: r for r in src_runs}

    overall_summary: dict = {}
    overall_ct = {"true_true": 0, "false_true": 0, "true_false": 0, "false_false": 0}
    puzzles_data: dict = {}

    for a in analyses:
        src = run_map.get(a.marc_run_id)
        if not src or not src.puzzle:
            continue

        pid   = src.puzzle.id
        ikey  = a.instruct_type or "N/A"
        ckey  = a.coding_type   or "N/A"
        passed = bool(src.did_pass)

        # Overall summary
        overall_summary.setdefault(ikey, {}).setdefault(
            ckey, {"total": 0, "passed": 0, "used": 0, "confidence_sum": 0.0})
        od = overall_summary[ikey][ckey]
        od["total"] += 1; od["confidence_sum"] += a.confidence_score
        if a.was_metaphor_used: od["used"] += 1
        if passed: od["passed"] += 1

        # Overall contingency
        ck = ("true" if a.was_metaphor_used else "false") + "_" + ("true" if passed else "false")
        overall_ct[ck] += 1

        # Per-puzzle
        if pid not in puzzles_data:
            puzzles_data[pid] = {
                "metaphor": src.puzzle.metaphor or "",
                "summary": {}, "runs": [],
                "ct": {"true_true": 0, "false_true": 0, "true_false": 0, "false_false": 0},
            }
        pd_ = puzzles_data[pid]
        pd_["runs"].append({
            "run_id": a.marc_run_id, "instruct_type": ikey, "coding_type": ckey,
            "passed": passed, "metaphor_used": a.was_metaphor_used,
            "confidence": round(a.confidence_score, 3),
            "judge_reasoning": a.judge_reasoning or "",
        })
        pd_["summary"].setdefault(ikey, {}).setdefault(
            ckey, {"total": 0, "passed": 0, "used": 0, "confidence_sum": 0.0})
        pd = pd_["summary"][ikey][ckey]
        pd["total"] += 1; pd["confidence_sum"] += a.confidence_score
        if a.was_metaphor_used: pd["used"] += 1
        if passed: pd["passed"] += 1
        pd_["ct"][ck] += 1

    return overall_summary, overall_ct, puzzles_data

def page_judge():
    st.header("🔍 Judge — Metaphor Analysis")
    j_url   = cfg.get("judge_api_url", "https://mindrouter.uidaho.edu/api/chat")
    j_model = cfg.get("judge_model", "openai/gpt-oss-120b")
    st.caption(f"Judge model: `{j_model}` · `{j_url}`  _(change in ⚙️ Settings → Judge & VLLM)_")

    tab_run, tab_view = st.tabs(["▶ Run Analysis", "📋 View Results"])

    with tab_run:
        st.subheader("Run LLM Judge on Reasoning Traces")
        cl, cr = st.columns(2)
        with cl:
            src_db = _source_db_selector("judge_src", "Source database")
            if src_db and src_db.exists():
                analysis_db = src_db.parent / f"{src_db.stem}_analysis.db"
                st.caption(f"Output → `{analysis_db.name}`")
            else:
                analysis_db = None
                if src_db: st.warning("Source DB not found.")
            pid_filter = st.number_input("Puzzle ID filter (0 = all)", 0, value=0, step=1, key="judge_pid")
        with cr:
            st.info("API settings are configured in **⚙️ Settings → Judge & VLLM**.")

        st.divider()
        go = st.button("▶ Run Judge", type="primary",
                       disabled=not (src_db and src_db.exists() and analysis_db is not None))

        if go:
            a_engine = create_engine(f"sqlite:///{analysis_db}")
            JudgeBase.metadata.create_all(a_engine)
            a_sess = sessionmaker(bind=a_engine)()
            s_sess = _src_session(src_db)
            try:
                processed = set(a_sess.execute(select(MetaphorAnalysis.marc_run_id)).scalars().all())
                st.info(f"Already analyzed: {len(processed)} runs")
                q = (s_sess.query(SrcRun).join(SrcPuzzle)
                     .filter(SrcPuzzle.metaphor != None, SrcPuzzle.metaphor != "",
                             SrcRun.id.notin_(processed))
                     .options(joinedload(SrcRun.puzzle)))
                if pid_filter > 0:
                    q = q.filter(SrcPuzzle.id == pid_filter)
                runs = q.all()
                if not runs:
                    st.info("No new runs to analyze.")
                else:
                    prog = st.progress(0); stat = st.empty(); ok = fail = 0
                    for i, run in enumerate(runs):
                        stat.text(f"Run {run.id} ({i+1}/{len(runs)})…")
                        result = run_judge(run.reasoning_text, run.puzzle.metaphor, j_url, j_model)
                        if result:
                            a_sess.add(MetaphorAnalysis(
                                marc_run_id=run.id, judge_model=j_model,
                                was_metaphor_used=result["was_metaphor_used"],
                                confidence_score=result["confidence_score"],
                                judge_reasoning=result["judge_reasoning"],
                                coding_type=run.coding_type, instruct_type=run.instruct_type))
                            a_sess.commit(); ok += 1
                        else:
                            fail += 1
                        prog.progress((i + 1) / len(runs)); time.sleep(0.5)
                    stat.text("Done!")
                    st.success(f"Analyzed {ok} runs. {fail} failed.")
            finally:
                s_sess.close(); a_sess.close()

    with tab_view:
        st.subheader("Analysis Results")
        src_db_r = _source_db_selector("judge_res", "Source database")
        a_db_r   = (src_db_r.parent / f"{src_db_r.stem}_analysis.db") if src_db_r and src_db_r.exists() else None

        if not a_db_r or not a_db_r.exists():
            st.info("No analysis database found. Run analysis first.")
            return

        a_sess_r = sessionmaker(bind=create_engine(f"sqlite:///{a_db_r}"))()
        s_sess_r = _src_session(src_db_r)
        try:
            overall_summary, overall_ct, puzzles_data = _build_judge_report_data(a_sess_r, s_sess_r)
            if overall_summary is None:
                st.info("No results yet."); return

            total_analyzed = sum(d["total"] for id_ in overall_summary.values() for d in id_.values())
            total_used     = sum(d["used"]   for id_ in overall_summary.values() for d in id_.values())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed",  total_analyzed)
            c2.metric("Metaphor Used",   total_used)
            c3.metric("Usage Rate", f"{total_used/total_analyzed*100:.1f}%" if total_analyzed else "—")

            st.divider()
            st.subheader("Overall Summary by Group")
            df_overall = _judge_summary_table(overall_summary)
            if not df_overall.empty:
                st.dataframe(df_overall.set_index(["Instruct Type", "Coding Type"]),
                             use_container_width=True)

            st.subheader("Overall Pass × Metaphor Correlation")
            _contingency_table(overall_ct)

            st.divider()
            st.subheader("Per-Puzzle Breakdown")
            for pid, pdata in sorted(puzzles_data.items()):
                with st.expander(f"🧩 Puzzle {pid}  —  {len(pdata['runs'])} runs analyzed"):
                    if pdata["metaphor"]:
                        st.info(f"**Metaphor:** {pdata['metaphor']}")

                    col_sum, col_ct = st.columns([3, 2])
                    with col_sum:
                        st.markdown("**Summary by Group**")
                        df_p = _judge_summary_table(pdata["summary"])
                        if not df_p.empty:
                            st.dataframe(df_p.set_index(["Instruct Type", "Coding Type"]),
                                         use_container_width=True)
                    with col_ct:
                        _contingency_table(pdata["ct"])

                    st.markdown("**All Runs**")
                    runs_by_instruct: dict = defaultdict(list)
                    for r in pdata["runs"]:
                        runs_by_instruct[r["instruct_type"]].append(r)

                    for instruct, instruct_runs in sorted(runs_by_instruct.items()):
                        with st.expander(f"{instruct} ({len(instruct_runs)} runs)"):
                            for r in instruct_runs:
                                ui = "✅" if r["metaphor_used"] else "❌"
                                pi = "✅" if r["passed"]        else "❌"
                                lc, rc = st.columns([1, 2])
                                with lc:
                                    st.markdown(f"**Run {r['run_id']}** · `{r['coding_type']}`")
                                    st.markdown(f"Pass: {pi}  |  Metaphor: {ui}  |  conf: `{r['confidence']}`")
                                with rc:
                                    st.text(r["judge_reasoning"])
                                st.divider()
        finally:
            a_sess_r.close(); s_sess_r.close()

# ---------------------------------------------------------------------------
# PAGE: Marcability
# ---------------------------------------------------------------------------

def page_marcability():
    st.header("📈 Marcability — Pass Rate Statistics")

    src_db = _source_db_selector("marc", "Source database")
    go = st.button("▶ Compute Pass Rates", type="primary",
                   disabled=not (src_db and src_db.exists()))

    if "marc_data" not in st.session_state:
        st.session_state["marc_data"] = None

    if go and src_db and src_db.exists():
        with st.spinner("Computing…"):
            try:
                st.session_state["marc_data"] = compute_marcability(src_db)
                st.success(f"Computed stats for {len(st.session_state['marc_data'])} puzzles.")
            except Exception as e:
                st.error(f"Error: {e}")

    data = st.session_state.get("marc_data")
    if not data:
        st.info("Select a database and click Compute Pass Rates.")
        return

    def _diff(row): return (row["metaphor_rate"] or 0) - (row["none_rate"] or 0)
    data = sorted(data, key=_diff, reverse=True)
    total = len(data)

    # Top-line metrics
    m_avg = sum(r["metaphor_rate"] or 0 for r in data) / total if total else 0
    n_avg = sum(r["none_rate"]     or 0 for r in data) / total if total else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Puzzles",               total)
    c2.metric("Avg Metaphor Pass Rate", f"{m_avg:.1%}")
    c3.metric("Avg No-Hint Pass Rate",  f"{n_avg:.1%}")

    # Chart
    st.subheader("Pass Rates by Puzzle")
    chart_rows = []
    for r in data:
        short = " ".join(r["metaphor"].split()[:4]) if r["metaphor"] else f"Puzzle {r['puzzle_id']}"
        if r["metaphor_rate"] is not None:
            chart_rows.append({"Puzzle": short, "Rate": r["metaphor_rate"], "Type": "Metaphor"})
        if r["none_rate"] is not None:
            chart_rows.append({"Puzzle": short, "Rate": r["none_rate"], "Type": "No Hint"})
    if chart_rows:
        fig = px.bar(chart_rows, x="Puzzle", y="Rate", color="Type", barmode="group",
                     range_y=[0, 1], labels={"Rate": "Pass Rate"},
                     color_discrete_map={"Metaphor": "#0074D9", "No Hint": "#AAAAAA"})
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    # Styled summary table (like the HTML report)
    st.subheader("Results Table")
    st.caption("Sorted by Metaphor − No-Hint rate difference. Green = metaphor helped, red = metaphor hurt.")

    table_rows = []
    for row in data:
        d     = _diff(row)
        short = " ".join(row["metaphor"].split()[:5]) if row["metaphor"] else ""
        # Build breakdown strings
        def bd_str(bd):
            return "  ".join(f"{ct}: {v}" for ct, v in bd.items()) if bd else "—"
        table_rows.append({
            "ID":       row["puzzle_id"],
            "Metaphor (first 5 words)": short,
            "Metaphor Rate": row["metaphor_rate"] if row["metaphor_rate"] is not None else float("nan"),
            "No-Hint Rate":  row["none_rate"]     if row["none_rate"]     is not None else float("nan"),
            "Δ (M − N)":     d,
            "M Runs":  row["metaphor_count"],
            "N Runs":  row["none_count"],
            "M Breakdown": bd_str(row["metaphor_breakdown"]),
            "N Breakdown": bd_str(row["none_breakdown"]),
        })

    df = pd.DataFrame(table_rows)

    def _color_diff(val):
        if pd.isna(val): return ""
        if val > 0:  return "color: #16a34a; font-weight: bold"
        if val < 0:  return "color: #dc2626; font-weight: bold"
        return "color: #6b7280"

    def _fmt_pct(val):
        return f"{val:.1%}" if not pd.isna(val) else "—"

    styled = (
        df.style
        .format({
            "Metaphor Rate": _fmt_pct,
            "No-Hint Rate":  _fmt_pct,
            "Δ (M − N)":     lambda v: (f"+{v:.1%}" if v > 0 else f"{v:.1%}") if not pd.isna(v) else "—",
        })
        .applymap(_color_diff, subset=["Δ (M − N)"])
        .set_properties(**{"text-align": "left"})
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)

    # Per-puzzle full metaphor + breakdown detail
    st.subheader("Puzzle Detail")
    for row in data:
        d = _diff(row)
        mr, nr = row["metaphor_rate"], row["none_rate"]
        short  = " ".join(row["metaphor"].split()[:5]) if row["metaphor"] else f"Puzzle {row['puzzle_id']}"
        parts  = [f"Puzzle {row['puzzle_id']} · {short}"]
        if mr is not None: parts.append(f"Metaphor {mr:.1%} ({row['metaphor_count']}n)")
        if nr is not None: parts.append(f"None {nr:.1%} ({row['none_count']}n)")
        if mr is not None and nr is not None:
            sign = "+" if d >= 0 else ""
            parts.append(f"Δ {sign}{d:.1%}")
        with st.expander("  |  ".join(parts)):
            lc, rc = st.columns(2)
            with lc:
                st.markdown("**Metaphor breakdown by coding type:**")
                for ct, score in (row["metaphor_breakdown"] or {}).items():
                    st.caption(f"  `{ct}`: {score}")
                if not row["metaphor_breakdown"]: st.caption("No runs")
            with rc:
                st.markdown("**No-hint breakdown by coding type:**")
                for ct, score in (row["none_breakdown"] or {}).items():
                    st.caption(f"  `{ct}`: {score}")
                if not row["none_breakdown"]: st.caption("No runs")
            if row["metaphor"]:
                st.info(f"**Full metaphor:** {row['metaphor']}")

    # CSV download
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(csv_buf, fieldnames=[
        "Puzzle ID", "Puzzle Name", "Metaphor Pass Rate", "None Pass Rate",
        "Metaphor Run Count", "None Run Count", "Details"])
    writer.writeheader()
    for row in data:
        short5 = " ".join(row["metaphor"].split()[:5]) if row["metaphor"] else ""
        writer.writerow({
            "Puzzle ID":          row["puzzle_id"],
            "Puzzle Name":        short5,
            "Metaphor Pass Rate": f"{row['metaphor_rate']:.4f}" if row["metaphor_rate"] is not None else "",
            "None Pass Rate":     f"{row['none_rate']:.4f}"     if row["none_rate"]     is not None else "",
            "Metaphor Run Count": row["metaphor_count"],
            "None Run Count":     row["none_count"],
            "Details": json.dumps({"metaphor": row["metaphor_breakdown"],
                                   "none": row["none_breakdown"]}),
        })
    st.download_button("⬇️ Download CSV", csv_buf.getvalue(), "pass_rates.csv", "text/csv")

# ---------------------------------------------------------------------------
# PAGE: Strategies
# ---------------------------------------------------------------------------

def page_strategies():
    st.header("🧠 Strategies — Extraction & Viewer")
    j_url   = cfg.get("judge_api_url", "https://mindrouter.uidaho.edu/api/chat")
    j_model = cfg.get("judge_model", "openai/gpt-oss-120b")
    st.caption(f"Judge model: `{j_model}` · `{j_url}`  _(change in ⚙️ Settings → Judge & VLLM)_")

    tab_run, tab_view = st.tabs(["▶ Extract Strategies", "📋 View Strategies"])

    with tab_run:
        st.subheader("Extract Strategies from Successful Runs")
        cl, cr = st.columns(2)
        with cl:
            src_db = _source_db_selector("strat_src", "Source database")
            if src_db and src_db.exists():
                strat_db = src_db.parent / f"{src_db.stem}_STRATEGIES.db"
                st.caption(f"Output → `{strat_db.name}`")
            else:
                strat_db = None
                if src_db: st.warning("Source DB not found.")
            pid_filter = st.number_input("Puzzle ID filter (0 = all)", 0, value=0, step=1, key="strat_pid")
        with cr:
            st.info("API settings are configured in **⚙️ Settings → Judge & VLLM**.")

        st.divider()
        go = st.button("▶ Extract Strategies", type="primary",
                       disabled=not (src_db and src_db.exists() and strat_db is not None))

        if go:
            st_engine = create_engine(f"sqlite:///{strat_db}")
            StratBase.metadata.create_all(st_engine)
            st_sess = sessionmaker(bind=st_engine)()
            s_sess  = _src_session(src_db)
            try:
                processed = set(st_sess.execute(select(StrategyAnalysis.marc_run_id)).scalars().all())
                st.info(f"Already extracted: {len(processed)} runs")
                q = (s_sess.query(SrcRun)
                     .options(joinedload(SrcRun.puzzle))
                     .filter(SrcRun.did_pass == True, SrcRun.id.notin_(processed)))
                if pid_filter > 0:
                    q = q.filter(SrcRun.puzzle_id == pid_filter)
                runs = q.all()
                if not runs:
                    st.info("No new successful runs to process.")
                else:
                    prog = st.progress(0); stat = st.empty(); ok = 0
                    for i, run in enumerate(runs):
                        stat.text(f"Run {run.id} ({i+1}/{len(runs)})…")
                        result = run_strategy_extraction(run.reasoning_text, j_url, j_model)
                        if result:
                            st_sess.add(StrategyAnalysis(
                                marc_run_id=run.id, puzzle_id=run.puzzle_id,
                                metaphor=run.puzzle.metaphor if run.puzzle else None,
                                instruct_type=run.instruct_type, coding_type=run.coding_type,
                                judge_model=j_model,
                                attempted_strategies=json.dumps(result["attempted_strategies"]),
                                winning_strategy=result["winning_strategy"]))
                            st_sess.commit(); ok += 1
                        prog.progress((i + 1) / len(runs)); time.sleep(0.5)
                    stat.text("Done!")
                    st.success(f"Extracted strategies for {ok}/{len(runs)} runs.")
            finally:
                s_sess.close(); st_sess.close()

    with tab_view:
        st.subheader("Extracted Strategies")
        src_db_v  = _source_db_selector("strat_view", "Source database")
        strat_db_v = (src_db_v.parent / f"{src_db_v.stem}_STRATEGIES.db") if src_db_v and src_db_v.exists() else None

        if not strat_db_v or not strat_db_v.exists():
            st.info("No strategies database found. Run extraction first.")
            return

        st_sess_v = sessionmaker(bind=create_engine(f"sqlite:///{strat_db_v}"))()
        try:
            entries = st_sess_v.execute(select(StrategyAnalysis)).scalars().all()
            if not entries:
                st.info("No strategies extracted yet."); return

            # Summary metrics
            st.metric("Total Strategies Extracted", len(entries))
            puzzle_ids     = sorted({e.puzzle_id    for e in entries})
            instruct_types = sorted({e.instruct_type for e in entries})

            ff1, ff2 = st.columns(2)
            with ff1: f_p = st.multiselect("Puzzle ID",     puzzle_ids,     default=puzzle_ids,     key="sv_p")
            with ff2: f_i = st.multiselect("Instruct Type", instruct_types, default=instruct_types, key="sv_i")

            filtered = [e for e in entries if e.puzzle_id in f_p and e.instruct_type in f_i]
            st.caption(f"Showing {len(filtered)} of {len(entries)}")

            # Summary table: how many winning strategies per puzzle
            st.subheader("Winning Strategies Summary")
            summary_rows = []
            grouped: dict = defaultdict(list)
            for e in filtered: grouped[e.puzzle_id].append(e)
            for pid, group in sorted(grouped.items()):
                metaphor = group[0].metaphor or ""
                winning_strategies = list({e.winning_strategy for e in group})
                summary_rows.append({
                    "Puzzle ID": pid,
                    "Metaphor (preview)": " ".join(metaphor.split()[:6]),
                    "Successful Runs": len(group),
                    "Unique Winning Strategies": len(winning_strategies),
                })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows).set_index("Puzzle ID"),
                             use_container_width=True)

            # Per-puzzle detail
            st.subheader("Detail by Puzzle")
            for pid, group in sorted(grouped.items()):
                with st.expander(f"🧩 Puzzle {pid} — {len(group)} successful run(s)"):
                    if group[0].metaphor:
                        st.info(f"**Metaphor:** {group[0].metaphor}")

                    for e in group:
                        strategies = json.loads(e.attempted_strategies) if e.attempted_strategies else []
                        lc, rc = st.columns([1, 2])
                        with lc:
                            st.markdown(f"**Run {e.marc_run_id}** · `{e.instruct_type}` · `{e.coding_type}`")
                        with rc:
                            if strategies:
                                st.markdown("Attempted:")
                                for s in strategies: st.markdown(f"  - {s}")
                        st.success(f"✅ **Winning strategy:** {e.winning_strategy}")
                        st.divider()
        finally:
            st_sess_v.close()

# ---------------------------------------------------------------------------
# PAGE: VLLM
# ---------------------------------------------------------------------------

def page_vllm():
    st.header("🖼 VLLM — Visual LLM Experiments")
    vllm_url   = cfg.get("vllm_api_url", "https://mindrouter.uidaho.edu/api/chat")
    vllm_model = cfg.get("vllm_model", "qwen3-VL-32k:32b")
    st.caption(f"VLLM model: `{vllm_model}` · `{vllm_url}`  _(change in ⚙️ Settings → Judge & VLLM)_")

    tab_gen, tab_run, tab_results = st.tabs(
        ["🎨 Generate Images", "🤖 Run Experiment", "📋 Results"])

    with tab_gen:
        st.subheader("Generate Puzzle Images")
        cl, cr = st.columns(2)
        with cl:
            src_mode = st.radio("Source", ["Directory", "Upload"], horizontal=True, key="vg_mode")
            puzzle_files = []
            if src_mode == "Directory":
                default_dir = str(BASE_DIR / "MARCF25Comp")
                if not Path(default_dir).is_dir(): default_dir = str(PUZZLES_DIR)
                dir_path = st.text_input("Puzzle directory", value=default_dir, key="vg_dir")
                if dir_path.strip() and Path(dir_path.strip()).is_dir():
                    puzzle_files = sorted(Path(dir_path.strip()).glob("*.json"))
                    st.success(f"Found {len(puzzle_files)} puzzle(s)")
                elif dir_path.strip():
                    st.error("Directory not found")
            else:
                up = st.file_uploader("Upload puzzle JSON", type=["json"], key="vg_up")
                if up:
                    try:
                        data = json.loads(up.read())
                        st.session_state["_vllm_up"] = {"name": up.name, "data": data}
                        puzzle_files = ["__uploaded__"]
                        st.success(f"Loaded: {up.name}")
                    except Exception as e:
                        st.error(f"Parse error: {e}")
        with cr:
            show_solution = st.toggle("Show test solution (answered)", value=False)
            st.caption("**Off** = unanswered images for VLLM input (test output hidden).\n\n**On** = answered images showing ground truth.")

        gen_btn = st.button("🎨 Generate Images", type="primary", disabled=not puzzle_files)
        if gen_btn and puzzle_files:
            generated = []
            prog = st.progress(0)
            for i, pf in enumerate(puzzle_files):
                if pf == "__uploaded__":
                    up_d = st.session_state.get("_vllm_up", {})
                    name, pdata = up_d.get("name", "puzzle.json"), up_d.get("data", {})
                else:
                    name = pf.name
                    try:
                        pdata = json.loads(pf.read_text(encoding="utf-8"))
                    except Exception as e:
                        st.warning(f"Could not read {name}: {e}")
                        prog.progress((i + 1) / len(puzzle_files)); continue
                img_bytes = visualize_puzzle_to_bytes(pdata, show_solution=show_solution)
                if img_bytes:
                    generated.append({"name": name, "bytes": img_bytes})
                prog.progress((i + 1) / len(puzzle_files))
            st.session_state["vllm_generated"] = generated
            st.success(f"Generated {len(generated)} image(s).")

        generated = st.session_state.get("vllm_generated", [])
        if generated:
            st.subheader(f"Generated Images ({len(generated)})")
            for row_i in range(0, len(generated), 2):
                cols = st.columns(2)
                for col, item in zip(cols, generated[row_i:row_i+2]):
                    with col:
                        st.caption(f"**{item['name']}**")
                        st.image(item["bytes"], use_container_width=True)
                        st.download_button("⬇️ Download PNG", item["bytes"],
                                           Path(item["name"]).stem + ".png", "image/png",
                                           key=f"vdl_{item['name']}")

    with tab_run:
        st.subheader("Run Multimodal LLM on a Puzzle Image")
        generated = st.session_state.get("vllm_generated", [])

        cl, cr = st.columns([1, 1])
        with cl:
            selected_img: Optional[bytes] = None
            if generated:
                img_names = [g["name"] for g in generated]
                sel = st.selectbox("Use a generated image", img_names, key="vr_sel")
                selected_img = next((g["bytes"] for g in generated if g["name"] == sel), None)
                selected_name = sel
                if selected_img: st.image(selected_img, use_container_width=True)
            else:
                st.info("Generate images first, or upload one below.")
                selected_name = None
            up_img = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"], key="vr_up")
            if up_img:
                selected_img  = up_img.read()
                selected_name = up_img.name
                st.image(selected_img, use_container_width=True)
        with cr:
            custom_prompt = st.text_area("Prompt", value=VLLM_PROMPT.strip(), height=250, key="vr_prompt")

        st.divider()
        run_btn = st.button("🤖 Run Experiment", type="primary", disabled=selected_img is None)

        if run_btn and selected_img:
            with st.spinner(f"Sending to `{vllm_model}`… (may take several minutes)"):
                result = run_vllm(selected_img, vllm_url, vllm_model, prompt=custom_prompt)

            # Save to DB regardless of outcome
            try:
                vs = _get_vllm_session()
                vs.add(VLLMRun(
                    puzzle_name  = selected_name,
                    model        = vllm_model,
                    api_url      = vllm_url,
                    reasoning    = result["reasoning"],
                    output_grid  = json.dumps(result["output_grid"]) if result["output_grid"] else None,
                    raw_response = result["raw"],
                ))
                vs.commit(); vs.close()
            except Exception as e:
                st.warning(f"Could not save run to DB: {e}")

            if result["error"]:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Response received!")
                rc1, rc2 = st.columns([2, 1])
                with rc1:
                    st.subheader("Reasoning")
                    st.text_area("", value=result["reasoning"], height=300, key="vr_out")
                with rc2:
                    st.subheader("Output Grid")
                    if result["output_grid"]:
                        b64 = grid_to_base64(result["output_grid"])
                        if b64: st.image(base64.b64decode(b64), use_container_width=True)
                        st.json(result["output_grid"])
                    else:
                        st.warning("No grid parsed.")
                        if result["raw"]:
                            st.text_area("Raw response", value=result["raw"], height=200, key="vr_raw")

    with tab_results:
        st.subheader("VLLM Experiment Results")
        if not VLLM_DB_PATH.exists():
            st.info("No VLLM runs recorded yet.")
            return

        vs = _get_vllm_session()
        try:
            runs = vs.execute(select(VLLMRun).order_by(VLLMRun.created_at.desc())).scalars().all()
            if not runs:
                st.info("No VLLM runs recorded yet."); return

            # Summary
            models  = sorted({r.model for r in runs})
            puzzles = sorted({r.puzzle_name or "unknown" for r in runs})
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Runs",     len(runs))
            c2.metric("Models Used",    len(models))
            c3.metric("Puzzles Tested", len(puzzles))

            # Filters
            ff1, ff2 = st.columns(2)
            with ff1: f_m = st.multiselect("Model",  models,  default=models,  key="vr_fm")
            with ff2: f_p = st.multiselect("Puzzle", puzzles, default=puzzles, key="vr_fp")

            filtered = [r for r in runs
                        if r.model in f_m and (r.puzzle_name or "unknown") in f_p]
            st.caption(f"Showing {len(filtered)} of {len(runs)} runs")

            # Group by puzzle
            grouped_runs: dict = defaultdict(list)
            for r in filtered: grouped_runs[r.puzzle_name or "unknown"].append(r)

            for puzzle_name, pruns in sorted(grouped_runs.items()):
                with st.expander(f"🖼 {puzzle_name}  —  {len(pruns)} run(s)"):
                    for run in pruns:
                        date_str = str(run.created_at)[:16] if run.created_at else ""
                        lc, mc, rc = st.columns([1, 1, 2])
                        with lc:
                            st.markdown(f"**Run {run.id}**")
                            st.caption(f"`{run.model}`")
                            st.caption(date_str)
                        with mc:
                            st.markdown("**Output Grid**")
                            if run.output_grid:
                                try:
                                    grid = json.loads(run.output_grid)
                                    if grid:
                                        b64 = grid_to_base64(grid)
                                        if b64: st.image(base64.b64decode(b64), use_container_width=True)
                                except Exception:
                                    st.caption("Could not render grid")
                            else:
                                st.caption("No grid")
                        with rc:
                            st.markdown("**Reasoning**")
                            st.text_area("", value=run.reasoning or "None",
                                         height=120, key=f"vr_rsn_{run.id}")
                        st.divider()
        finally:
            vs.close()

# ---------------------------------------------------------------------------
# PAGE: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.header("⚙️ Settings")
    cfg = get_cfg()
    tab_api, tab_prompts, tab_judge_vllm = st.tabs(
        ["🔌 API", "📝 Prompts", "🔍 Judge & VLLM"])

    with tab_api:
        st.subheader("Provider Profiles")
        st.caption(
            "Each provider saves its own API key and model. "
            "Use the **sidebar switcher** to activate a provider — it loads that profile instantly."
        )
        _provider_banner(cfg)

        profiles = cfg.get("provider_profiles", {k: dict(v) for k, v in DEFAULT_PROVIDER_PROFILES.items()})
        active_key = cfg.get("provider", "mindrouter")
        new_profiles = {}

        for display_name in PROVIDER_DISPLAY_ORDER:
            pkey    = display_name.lower()
            profile = profiles.get(pkey, {})
            is_active = (pkey == active_key)
            label = f"{PROVIDER_ICONS.get(pkey, '⚫')} {display_name}" + (" ← active" if is_active else "")
            with st.expander(label, expanded=is_active):
                st.text_input(
                    "Endpoint URL",
                    value=PROVIDER_URLS.get(pkey, ""),
                    key=f"p_url_{pkey}",
                    disabled=(pkey != "custom"),
                )
                p_key   = st.text_input("API Key",    value=profile.get("api_key", ""),    type="password", key=f"p_key_{pkey}")
                p_model = st.text_input("Model Name", value=profile.get("model_name", ""),                  key=f"p_model_{pkey}")
                if pkey == "mindrouter":
                    st.caption("Sends extra `format` + `think` fields for structured output.")
                elif pkey == "gemini":
                    st.caption("OpenAI-compatible endpoint. Key = Google AI Studio API key.")
                elif pkey == "claude":
                    st.caption("Uses native Anthropic Messages API with extended thinking. Key = Anthropic API key.")
                new_profiles[pkey] = {"api_key": p_key, "model_name": p_model}

        # Active provider values drive the top-level cfg fields
        _ap      = new_profiles.get(active_key, {})
        provider = active_key
        api_url  = PROVIDER_URLS.get(active_key, cfg.get("api_url", ""))
        api_key  = _ap.get("api_key", cfg["api_key"])
        model_name = _ap.get("model_name", cfg["model_name"])

        st.divider()
        st.caption("**Request Timeouts**")
        t1, t2 = st.columns(2)
        with t1: connect_timeout = st.number_input("Connect Timeout (s)", 1, 60,   int(cfg.get("connect_timeout", 10)))
        with t2: read_timeout    = st.number_input("Read Timeout (s)",    10, 3600, int(cfg.get("read_timeout", 180)))

        st.divider()
        st.caption("**Thinking Token Budget**")
        st.caption(
            "How many tokens each model may use for chain-of-thought reasoning. "
            "Applies to Claude (extended thinking), MindRouter (`think_budget`), and is "
            "recorded for every run so results across models are comparable."
        )
        thinking_budget = st.number_input(
            "Thinking Budget (tokens)", min_value=1000, max_value=100000,
            value=int(cfg.get("thinking_budget", 10000)), step=1000,
        )

    with tab_prompts:
        st.subheader("System Prompts")
        base_system = st.text_area("Base System Prompt", value=cfg["base_system_prompt"], height=100)
        cmt_system  = st.text_area("CMT System Prompt",  value=cfg["cmt_system_prompt"],  height=220)
        st.divider()
        st.subheader("User Prompt Templates")
        st.caption("Use `{metaphor}` as a placeholder where the metaphor hint should appear.")
        base_raw_instructions     = st.text_area("Base Raw — Instructions",                   value=cfg["base_raw_instructions"],     height=80)
        metaphor_hint_template    = st.text_area("Base Metaphor — Hint *(uses {metaphor})*",  value=cfg["metaphor_hint_template"],    height=100)
        cmt_raw_instructions      = st.text_area("CMT Raw — Instructions",                    value=cfg["cmt_raw_instructions"],      height=120)
        cmt_metaphor_instructions = st.text_area("CMT Metaphor *(uses {metaphor})*",          value=cfg["cmt_metaphor_instructions"], height=180)

    with tab_judge_vllm:
        st.subheader("Judge & Strategy Extraction")
        st.caption("Used by 🔍 Judge and 🧠 Strategies pages.")
        judge_api_url = st.text_input("Judge API URL",  value=cfg.get("judge_api_url", "https://mindrouter.uidaho.edu/api/chat"))
        judge_model   = st.text_input("Judge Model",    value=cfg.get("judge_model",   "openai/gpt-oss-120b"))
        st.divider()
        st.subheader("VLLM Experiments")
        st.caption("Used by 🖼 VLLM page.")
        vllm_api_url  = st.text_input("VLLM API URL",  value=cfg.get("vllm_api_url", "https://mindrouter.uidaho.edu/api/chat"))
        vllm_model    = st.text_input("VLLM Model",    value=cfg.get("vllm_model",   "qwen3-VL-32k:32b"))

    st.divider()
    sc, rc = st.columns([1, 4])
    with sc:
        if st.button("💾 Save Settings", type="primary"):
            new_cfg = dict(cfg)
            new_cfg.update({
                "provider": provider, "api_url": api_url, "api_key": api_key,
                "model_name": model_name, "connect_timeout": connect_timeout, "read_timeout": read_timeout,
                "thinking_budget": thinking_budget,
                "provider_profiles": new_profiles,
                "base_system_prompt": base_system, "cmt_system_prompt": cmt_system,
                "base_raw_instructions": base_raw_instructions,
                "metaphor_hint_template": metaphor_hint_template,
                "cmt_raw_instructions": cmt_raw_instructions,
                "cmt_metaphor_instructions": cmt_metaphor_instructions,
                "judge_api_url": judge_api_url, "judge_model": judge_model,
                "vllm_api_url": vllm_api_url,   "vllm_model":  vllm_model,
            })
            save_config(new_cfg)
            st.session_state["cfg"] = new_cfg
            st.success("Settings saved!")
    with rc:
        if st.button("↺ Reset to Defaults"):
            save_config(DEFAULT_CONFIG)
            st.session_state["cfg"] = dict(DEFAULT_CONFIG)
            st.success("Reset to defaults."); st.rerun()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if   page == "🧪 BAFL":         page_bafl()
elif page == "🔍 Judge":        page_judge()
elif page == "📈 Marcability":  page_marcability()
elif page == "🧠 Strategies":   page_strategies()
elif page == "🖼 VLLM":         page_vllm()
elif page == "⚙️ Settings":     page_settings()



