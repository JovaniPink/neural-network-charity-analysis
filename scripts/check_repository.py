#!/usr/bin/env python3
"""Validate the neural-network archive without executing or loading artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
NOTEBOOKS = (
    "01-Keras-Intro.ipynb",
    "02-Ramen-Ratings.ipynb",
    "03-Standardize.ipynb",
    "04-DeepLearning-Tabular.ipynb",
    "AlphabetSoupCharity-Optimzation.ipynb",
    "AlphabetSoupCharity.ipynb",
)
CSV_SCHEMAS = {
    "data/HR-Employee-Attrition.csv": (
        "Age",
        "Attrition",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EmployeeCount",
        "EmployeeNumber",
        "EnvironmentSatisfaction",
        "Gender",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "Over18",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StandardHours",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ),
    "data/charity_data.csv": (
        "EIN",
        "NAME",
        "APPLICATION_TYPE",
        "AFFILIATION",
        "CLASSIFICATION",
        "USE_CASE",
        "ORGANIZATION",
        "STATUS",
        "INCOME_AMT",
        "SPECIAL_CONSIDERATIONS",
        "ASK_AMT",
        "IS_SUCCESSFUL",
    ),
    "data/hr_dataset.csv": (
        "Satisfaction_Level",
        "Num_Projects",
        "Time_Spent",
        "Num_Promotions",
    ),
    "data/ramen-ratings.csv": (
        "Review #",
        "Brand",
        "Variety",
        "Style",
        "Country",
        "Stars",
        "Top Ten",
    ),
}
MODEL_DIRECTORIES = (
    "models/model-001",
    "models/model-002",
    "models/model-005",
    "models/model-007",
    "models/model-009",
    "models/model-010",
    "models/model-011",
    "models/model-012",
    "models/model-013",
    "models/model-014",
    "models/model-015",
    "models/model-017",
    "models/model-019",
    "models/model-020",
    "models/model-023",
    "models/model-024",
    "models/model-034",
    "models/model-083",
    "models/model-checkpoints-color/model",
)
DISALLOWED_SERIALIZED_SUFFIXES = {".h5", ".hdf5", ".joblib", ".pickle", ".pkl"}


class ValidationError(RuntimeError):
    """Raised when a repository invariant is not satisfied."""


def load_notebook(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing notebook: {path.relative_to(ROOT)}")
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid notebook JSON in {path.name}: {exc}") from exc
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValidationError(f"{path.name} is not notebook format 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValidationError(f"{path.name} has no notebook cells")
    return notebook


def validate_csv(relative_path: str, expected_header: tuple[str, ...]) -> int:
    path = ROOT / relative_path
    if not path.is_file():
        raise ValidationError(f"missing CSV snapshot: {relative_path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
            next(reader)
        except StopIteration as exc:
            raise ValidationError(f"CSV snapshot has no data rows: {relative_path}") from exc
    if header != expected_header:
        raise ValidationError(f"unexpected CSV schema in {relative_path}: {header!r}")
    return path.stat().st_size


def validate_model_directory(relative_path: str) -> None:
    directory = ROOT / relative_path
    required = (
        directory / "saved_model.pb",
        directory / "variables" / "variables.index",
        directory / "variables" / "variables.data-00000-of-00001",
    )
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        raise ValidationError(
            f"incomplete SavedModel {relative_path}: "
            + ", ".join(str(path) for path in missing)
        )


def validate_inventory_and_sizes() -> None:
    discovered_models = tuple(
        sorted(
            str(path.parent.relative_to(ROOT))
            for path in (ROOT / "models").rglob("saved_model.pb")
        )
    )
    if discovered_models != tuple(sorted(MODEL_DIRECTORIES)):
        raise ValidationError(
            "SavedModel inventory changed; review and update the explicit contract"
        )

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.casefold() in DISALLOWED_SERIALIZED_SUFFIXES:
            raise ValidationError(
                f"unexpected serialized model format: {path.relative_to(ROOT)}"
            )
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            raise ValidationError(
                f"{path.relative_to(ROOT)} is {size} bytes; limit is {MAX_TRACKED_FILE_BYTES}"
            )


def main() -> int:
    try:
        validate_inventory_and_sizes()
        notebook_results = []
        for name in NOTEBOOKS:
            notebook = load_notebook(ROOT / name)
            notebook_results.append((name, len(notebook["cells"])))
        csv_sizes = {
            path: validate_csv(path, schema) for path, schema in CSV_SCHEMAS.items()
        }
        for path in MODEL_DIRECTORIES:
            validate_model_directory(path)
    except ValidationError as exc:
        print(f"archive validation failed: {exc}", file=sys.stderr)
        return 1

    print("archive validation passed")
    for name, cell_count in notebook_results:
        print(f"- {name}: notebook format 4, {cell_count} cells")
    print(f"- CSV snapshots: {len(csv_sizes)} expected schemas")
    print(f"- TensorFlow SavedModel artifacts: {len(MODEL_DIRECTORIES)} complete")
    print("- model deserialization: not performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
