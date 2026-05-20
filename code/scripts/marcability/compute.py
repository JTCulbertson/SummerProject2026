"""
scripts/marcability/compute.py — MARC Pass-Rate Statistics Export
=================================================================
Calculates per-puzzle pass rates broken down by instruction type (metaphor vs
none) and coding strategy (color, object, etc.), then exports the results to a
CSV file.

Usage:
    python scripts/marcability/compute.py <db_path>
    # e.g. python scripts/marcability/compute.py data/gpt5.db
    # Outputs: data/gpt5_pass_rates.csv
"""
import csv
import json
import os
import argparse
import logging
from pathlib import Path
from sqlalchemy import (
    Boolean, Column, Enum, Float, ForeignKey, Integer,
    String, Text, create_engine, select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, selectinload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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


def calculate_rate(runs):
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.did_pass) / len(runs)

def get_breakdown(runs):
    breakdown = {}
    runs_by_type = {}
    for r in runs:
        runs_by_type.setdefault(r.coding_type, []).append(r)
    for c_type, type_runs in runs_by_type.items():
        passed = sum(1 for r in type_runs if r.did_pass)
        breakdown[c_type] = f"{passed}/{len(type_runs)}"
    return breakdown

def get_short_name(metaphor_text):
    if not metaphor_text:
        return "No Metaphor Provided"
    return " ".join(metaphor_text.split()[:5])


def main():
    parser = argparse.ArgumentParser(description="Compute pass rates and export to CSV.")
    parser.add_argument("db", help="Path to the SQLite source database")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logging.error(f"Database file not found: {db_path}")
        return

    output_csv = db_path.parent / f"{db_path.stem}_pass_rates.csv"

    engine = create_engine(f"sqlite:///{db_path}")
    session = sessionmaker(bind=engine)()

    try:
        puzzles = session.execute(
            select(Puzzle).options(selectinload(Puzzle.runs))
        ).scalars().all()
        logging.info(f"Found {len(puzzles)} puzzles.")

        csv_data = []
        for puzzle in puzzles:
            metaphor_runs = [r for r in puzzle.runs if r.instruct_type == "metaphor"]
            none_runs     = [r for r in puzzle.runs if r.instruct_type == "none"]

            csv_data.append({
                "Puzzle ID":           puzzle.id,
                "Puzzle Name":         get_short_name(puzzle.metaphor),
                "Metaphor Pass Rate":  f"{calculate_rate(metaphor_runs):.2f}",
                "None Pass Rate":      f"{calculate_rate(none_runs):.2f}",
                "Metaphor Run Count":  len(metaphor_runs),
                "None Run Count":      len(none_runs),
                "Details":             json.dumps({
                    "metaphor": get_breakdown(metaphor_runs),
                    "none":     get_breakdown(none_runs),
                }),
            })

        fieldnames = [
            "Puzzle ID", "Puzzle Name",
            "Metaphor Pass Rate", "None Pass Rate",
            "Metaphor Run Count", "None Run Count", "Details",
        ]

        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        logging.info(f"Written to: {output_csv}")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        session.close()

if __name__ == "__main__":
    main()
