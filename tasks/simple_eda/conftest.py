"""
Conftest for the Employee Churn Pipeline task.

Generates employee_data.csv with opaque column names.

DATASET: 400 employees, 8 columns (var_01–var_08, shuffled).

PLANTED PROPERTIES:

  P1 — satisfaction is the strongest predictor of churn
       mean(satisfaction | churned=1) ≈ 4.0
       mean(satisfaction | churned=0) ≈ 7.0
       The relationship is monotone: lower satisfaction → higher P(churn).

  P2 — salary is BIMODAL (two workforce tiers)
       low tier  (55%): ~ N(42 000, 7 000)
       high tier (45%): ~ N(88 000, 12 000)
       High-salary employees churn less (salary_high → lower P(churn)).

  P3 — department effect on churn
       Engineering: lowest churn  (~20 %)
       Sales:       highest churn (~50 %)
       Support/HR:  mid-range     (~35 %)

  P4 — churned is the binary TARGET (values 0/1, ~35 % overall churn rate)
"""
import os
import numpy as np
import pandas as pd
import pytest as _pytest


_INTERNAL_NAMES = [
    "emp_id", "salary", "dept", "tenure_months",
    "churned", "satisfaction", "n_projects", "weekly_hours",
]


def _make_column_map(seed: int = 42) -> dict:
    """Return mapping: internal_name → opaque CSV name (var_01 … var_08)."""
    rng = np.random.default_rng(seed + 500)
    opaque = [f"var_{i:02d}" for i in range(1, len(_INTERNAL_NAMES) + 1)]
    shuffled = rng.permutation(opaque).tolist()
    return dict(zip(_INTERNAL_NAMES, shuffled))


_INT2OPAQUE = _make_column_map(seed=42)

# PLANTED: what agents must discover and encode in config.py
PLANTED = {
    # opaque → semantic mapping (the answer agents must write in COLUMN_MAP)
    "column_map":             {v: k for k, v in _INT2OPAQUE.items()},
    # all opaque column names present in employee_data.csv
    "all_opaque_cols":        list(_INT2OPAQUE.values()),
    # target
    "target_opaque":          _INT2OPAQUE["churned"],
    "target_semantic":        "churned",
    # key predictor (P1)
    "key_predictor_opaque":   _INT2OPAQUE["satisfaction"],
    "key_predictor_semantic": "satisfaction",
    # salary (P2)
    "salary_opaque":          _INT2OPAQUE["salary"],
    "salary_semantic":        "salary",
    # department (P3)
    "dept_opaque":            _INT2OPAQUE["dept"],
    "dept_semantic":          "dept",
    # expected feature splits
    "numeric_semantics":      ["salary", "tenure_months", "satisfaction",
                               "n_projects", "weekly_hours"],
    "categorical_semantics":  ["dept"],
    # dept churn ordering
    "lowest_churn_dept":      "Engineering",
    "highest_churn_dept":     "Sales",
}


def _build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400

    # emp_id — integer key, no analytical value
    emp_id = np.arange(1, n + 1)

    # dept — categorical with planted churn rates (P3)
    dept = rng.choice(
        ["Engineering", "Sales", "Support", "HR"],
        size=n, p=[0.30, 0.25, 0.25, 0.20],
    )

    # tenure_months — gamma-distributed, 0–72
    tenure_months = np.round(rng.gamma(3.0, 12.0, n)).astype(int)
    tenure_months = np.clip(tenure_months, 0, 72)

    # n_projects — 1–8 (weak predictor, mostly noise)
    n_projects = rng.integers(1, 9, n)

    # weekly_hours — roughly normal, 30–70 (noise)
    weekly_hours = np.round(rng.normal(45.0, 8.0, n), 1)
    weekly_hours = np.clip(weekly_hours, 30.0, 70.0)

    # salary — bimodal (P2): low tier vs high tier
    salary_group = rng.choice([0, 1], n, p=[0.55, 0.45])
    salary = np.where(
        salary_group == 0,
        rng.normal(42_000, 7_000, n),
        rng.normal(88_000, 12_000, n),
    )
    salary = np.round(np.clip(salary, 18_000, 140_000), -2).astype(int)

    # satisfaction — uniform 1–10; planted as the dominant churn driver (P1)
    satisfaction = np.round(rng.uniform(1.0, 10.0, n), 1)

    # churn probability — driven by satisfaction + salary + dept (P1, P2, P3)
    dept_adj = np.where(dept == "Engineering", -0.12,
               np.where(dept == "Sales",        +0.14,
               np.where(dept == "Support",       0.00,
                                                -0.04)))
    p_churn = (
        np.clip(0.78 - 0.07 * satisfaction, 0.03, 0.88)
        + 0.12 * (salary_group == 0)
        + dept_adj
    )
    p_churn = np.clip(p_churn, 0.02, 0.95)
    churned = (rng.random(n) < p_churn).astype(int)

    # Assemble with internal names, then rename to shuffled opaque names
    col_map = _make_column_map(seed=seed)
    internal_df = pd.DataFrame({
        "emp_id":        emp_id,
        "salary":        salary,
        "dept":          dept,
        "tenure_months": tenure_months,
        "churned":       churned,
        "satisfaction":  satisfaction,
        "n_projects":    n_projects,
        "weekly_hours":  weekly_hours,
    })
    df = internal_df.rename(columns=col_map)
    shuffled_order = (
        np.random.default_rng(seed + 2000).permutation(df.columns.tolist()).tolist()
    )
    return df[shuffled_order]


# ── pytest fixtures ────────────────────────────────────────────────────────────

@_pytest.fixture(scope="session", autouse=True)
def setup_dataset():
    """Write employee_data.csv to the repo root before tests run."""
    cwd = os.getcwd()
    df = _build_dataset(seed=42)
    df.to_csv(os.path.join(cwd, "employee_data.csv"), index=False)
    os.makedirs(os.path.join(cwd, "figures"), exist_ok=True)
    return {"df": df}


@_pytest.fixture(scope="session")
def ground_truth_df(setup_dataset):
    """Ground-truth DataFrame with opaque column names (as in employee_data.csv)."""
    return setup_dataset["df"]


@_pytest.fixture(scope="session")
def planted():
    """Planted-property constants for use in eval tests."""
    return PLANTED
