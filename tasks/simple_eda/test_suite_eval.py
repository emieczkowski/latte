"""
Private evaluation test suite — never copied into the agent repo.

Checks correctness of config.py and whether planted properties were discovered.

Test groups:
  A. TestOutputContract    — required files exist and are structurally valid
  B. TestConfigCompleteness — COLUMN_MAP covers all 8 columns; TARGET/FEATURES defined
  C. TestConfigAccuracy     — TARGET is actually binary; numeric/categorical split matches data
  D. TestPlantedFindings    — satisfaction → churn (P1), salary bimodal (P2), dept effect (P3)
  E. TestNarrative          — summary.txt is substantive and mentions the target variable
"""
import os
import sys
import json
import importlib.util
import pytest
import numpy as np
import pandas as pd

from conftest import PLANTED, _build_dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _cwd(*parts):
    return os.path.join(os.getcwd(), *parts)


def _load_config():
    path = _cwd("config.py")
    if not os.path.exists(path):
        pytest.fail("config.py not found in repo root.")
    spec = importlib.util.spec_from_file_location("_agent_config", path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.fail(f"config.py raised an error on import: {e}")
    return mod


def _load_findings():
    path = _cwd("findings.json")
    if not os.path.exists(path):
        pytest.fail("findings.json not found in repo root.")
    with open(path) as f:
        return json.load(f)


def _load_summary():
    path = _cwd("summary.txt")
    if not os.path.exists(path):
        pytest.fail("summary.txt not found in repo root.")
    return open(path).read()


def _renamed_df(cfg) -> pd.DataFrame:
    """Load employee_data.csv and apply COLUMN_MAP to get semantic column names."""
    df = pd.read_csv(_cwd("employee_data.csv"))
    col_map = getattr(cfg, "COLUMN_MAP", {})
    return df.rename(columns=col_map)


def _mentions(text: str, *keywords) -> bool:
    text = text.lower()
    return any(kw.lower() in text for kw in keywords)


def _findings_text(findings: dict, *categories) -> str:
    cats = categories or list(findings.keys())
    parts = []
    for cat in cats:
        for entry in findings.get(cat, []):
            if isinstance(entry, dict):
                parts.append(json.dumps(entry))
    return " ".join(parts)


def _entry_mentions_variable(entry: dict, *names) -> bool:
    text = json.dumps(entry).lower()
    return any(n.lower() in text for n in names)


# ── A. Output contract ────────────────────────────────────────────────────────

class TestOutputContract:
    def test_config_exists(self):
        assert os.path.exists(_cwd("config.py")), \
            "config.py must exist in the repo root"

    def test_config_importable(self):
        _load_config()

    def test_config_has_column_map(self):
        cfg = _load_config()
        assert hasattr(cfg, "COLUMN_MAP") and isinstance(cfg.COLUMN_MAP, dict), \
            "config.py must define COLUMN_MAP as a dict"

    def test_config_has_target(self):
        cfg = _load_config()
        assert hasattr(cfg, "TARGET") and isinstance(cfg.TARGET, str), \
            "config.py must define TARGET as a string"

    def test_config_has_feature_lists(self):
        cfg = _load_config()
        assert hasattr(cfg, "NUMERIC_FEATURES") and isinstance(cfg.NUMERIC_FEATURES, list), \
            "config.py must define NUMERIC_FEATURES as a list"
        assert hasattr(cfg, "CATEGORICAL_FEATURES") and isinstance(cfg.CATEGORICAL_FEATURES, list), \
            "config.py must define CATEGORICAL_FEATURES as a list"

    def test_findings_json_exists(self):
        assert os.path.exists(_cwd("findings.json"))

    def test_findings_json_valid(self):
        findings = _load_findings()
        required = {"distributions", "relationships", "subgroup_effects",
                    "outliers", "missing_data"}
        missing = required - set(findings.keys())
        assert not missing, f"findings.json missing keys: {missing}"

    def test_findings_categories_nonempty(self):
        findings = _load_findings()
        empty = [k for k in ("distributions", "relationships", "subgroup_effects",
                              "outliers", "missing_data")
                 if len(findings.get(k, [])) == 0]
        assert not empty, f"Empty finding categories: {empty}"

    def test_summary_exists(self):
        assert os.path.exists(_cwd("summary.txt"))


# ── B. Config completeness ────────────────────────────────────────────────────

class TestConfigCompleteness:
    def test_column_map_covers_all_columns(self):
        cfg = _load_config()
        col_map = cfg.COLUMN_MAP
        expected = set(PLANTED["all_opaque_cols"])
        mapped   = set(col_map.keys())
        missing  = expected - mapped
        assert not missing, (
            f"COLUMN_MAP is missing {len(missing)} opaque column(s): {missing}. "
            f"All 8 columns must be mapped."
        )

    def test_column_map_has_no_extra_keys(self):
        cfg = _load_config()
        expected = set(PLANTED["all_opaque_cols"])
        extra = set(cfg.COLUMN_MAP.keys()) - expected
        assert not extra, (
            f"COLUMN_MAP has unexpected keys (not in employee_data.csv): {extra}"
        )

    def test_column_map_values_are_unique(self):
        cfg = _load_config()
        vals = list(cfg.COLUMN_MAP.values())
        dupes = [v for v in set(vals) if vals.count(v) > 1]
        assert not dupes, \
            f"COLUMN_MAP maps multiple opaque columns to the same semantic name: {dupes}"

    def test_target_is_in_column_map(self):
        cfg = _load_config()
        assert cfg.TARGET in cfg.COLUMN_MAP.values(), (
            f"TARGET='{cfg.TARGET}' must be one of the values in COLUMN_MAP. "
            f"Current values: {list(cfg.COLUMN_MAP.values())}"
        )

    def test_numeric_features_in_column_map(self):
        cfg = _load_config()
        mapped_semantics = set(cfg.COLUMN_MAP.values())
        bad = [f for f in cfg.NUMERIC_FEATURES if f not in mapped_semantics]
        assert not bad, (
            f"NUMERIC_FEATURES contains names not in COLUMN_MAP values: {bad}"
        )

    def test_categorical_features_in_column_map(self):
        cfg = _load_config()
        mapped_semantics = set(cfg.COLUMN_MAP.values())
        bad = [f for f in cfg.CATEGORICAL_FEATURES if f not in mapped_semantics]
        assert not bad, (
            f"CATEGORICAL_FEATURES contains names not in COLUMN_MAP values: {bad}"
        )

    def test_no_feature_overlap(self):
        cfg = _load_config()
        overlap = set(cfg.NUMERIC_FEATURES) & set(cfg.CATEGORICAL_FEATURES)
        assert not overlap, \
            f"Columns in both NUMERIC_FEATURES and CATEGORICAL_FEATURES: {overlap}"


# ── C. Config accuracy ────────────────────────────────────────────────────────

class TestConfigAccuracy:
    def test_target_column_is_binary(self, ground_truth_df, planted):
        cfg = _load_config()
        df  = _renamed_df(cfg)
        target = cfg.TARGET
        assert target in df.columns, (
            f"After applying COLUMN_MAP, TARGET='{target}' is not a column. "
            f"Columns present: {list(df.columns)}"
        )
        unique_vals = set(df[target].dropna().unique())
        assert unique_vals <= {0, 1, True, False, 0.0, 1.0}, (
            f"TARGET column '{target}' should be binary (values 0/1), "
            f"found: {unique_vals}"
        )

    def test_numeric_features_are_numeric(self, ground_truth_df):
        cfg = _load_config()
        df  = _renamed_df(cfg)
        for col in cfg.NUMERIC_FEATURES:
            assert col in df.columns, \
                f"NUMERIC_FEATURES member '{col}' not found after applying COLUMN_MAP"
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' is listed in NUMERIC_FEATURES but has dtype "
                f"{df[col].dtype} — it is not numeric"
            )

    def test_categorical_features_are_not_numeric(self, ground_truth_df):
        cfg = _load_config()
        df  = _renamed_df(cfg)
        for col in cfg.CATEGORICAL_FEATURES:
            assert col in df.columns, \
                f"CATEGORICAL_FEATURES member '{col}' not found after applying COLUMN_MAP"
            assert not pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' is listed in CATEGORICAL_FEATURES but has numeric dtype "
                f"{df[col].dtype}"
            )

    def test_target_not_in_features(self):
        cfg = _load_config()
        all_features = set(cfg.NUMERIC_FEATURES) | set(cfg.CATEGORICAL_FEATURES)
        assert cfg.TARGET not in all_features, (
            f"TARGET='{cfg.TARGET}' must not appear in NUMERIC_FEATURES or "
            f"CATEGORICAL_FEATURES — it is the outcome, not a predictor"
        )

    def test_correct_target_column_identified(self, planted):
        """
        The column agents mapped to TARGET should be the actual churn indicator.
        We verify by checking that the opaque key for TARGET matches PLANTED['target_opaque'].
        """
        cfg = _load_config()
        # Find which opaque column maps to the agent's TARGET
        agent_target_opaque = None
        for opaque, semantic in cfg.COLUMN_MAP.items():
            if semantic == cfg.TARGET:
                agent_target_opaque = opaque
                break
        assert agent_target_opaque == planted["target_opaque"], (
            f"The column mapped to TARGET (opaque: '{agent_target_opaque}') is not the "
            f"actual churn column (expected opaque: '{planted['target_opaque']}'). "
            f"Examine the binary column — it has values 0/1 and ~35% positive rate."
        )


# ── D. Planted findings ───────────────────────────────────────────────────────

class TestPlantedFindings:
    def test_satisfaction_identified_as_predictor(self, planted):
        """
        P1: satisfaction is the dominant churn predictor.
        findings['relationships'] or ['subgroup_effects'] must reference it.
        """
        findings = _load_findings()
        sat_opaque   = planted["key_predictor_opaque"]
        sat_semantic = planted["key_predictor_semantic"]
        tgt_opaque   = planted["target_opaque"]
        tgt_semantic = planted["target_semantic"]

        relevant_entries = [
            e for cat in ("relationships", "subgroup_effects")
            for e in findings.get(cat, [])
            if isinstance(e, dict)
            and _entry_mentions_variable(e, sat_opaque, sat_semantic)
        ]
        assert relevant_entries, (
            f"No relationship entry mentions the satisfaction column "
            f"(opaque: '{sat_opaque}' or semantic: '{sat_semantic}'). "
            f"It is the strongest churn predictor — it must appear in "
            f"findings['relationships'] or findings['subgroup_effects']."
        )

    def test_satisfaction_churn_direction(self, ground_truth_df, planted):
        """
        P1 direction: low satisfaction → higher churn rate (verifiable from data).
        """
        sat_col = planted["key_predictor_opaque"]
        tgt_col = planted["target_opaque"]
        df = ground_truth_df.copy()
        low  = df[df[sat_col] <  5.0][tgt_col].mean()
        high = df[df[sat_col] >= 7.0][tgt_col].mean()
        assert low > high + 0.20, (
            f"Expected churn rate for low satisfaction (<5) to be at least 20pp "
            f"higher than high satisfaction (>=7). Got low={low:.3f}, high={high:.3f}. "
            f"Check dataset generation."
        )

    def test_satisfaction_finding_mentions_churn_link(self, planted):
        """
        The satisfaction finding should describe its link to churn, not just note it exists.
        """
        findings = _load_findings()
        sat_opaque   = planted["key_predictor_opaque"]
        sat_semantic = planted["key_predictor_semantic"]
        relevant_text = _findings_text(
            findings, "relationships", "subgroup_effects"
        )
        # Filter to entries that mention satisfaction
        relevant_entries = [
            e for cat in ("relationships", "subgroup_effects")
            for e in findings.get(cat, [])
            if isinstance(e, dict) and _entry_mentions_variable(e, sat_opaque, sat_semantic)
        ]
        combined = " ".join(json.dumps(e) for e in relevant_entries).lower()
        assert _mentions(combined,
            "churn", "left", "turnover", "attrition",
            "predict", "driver", "associated", "correlation",
            "negative", "lower satisfaction", "low satisfaction",
        ), (
            f"The satisfaction finding should describe its relationship to churn. "
            f"Found: {combined[:400]}"
        )

    def test_salary_bimodal_or_split_identified(self, planted):
        """
        P2: salary is bimodal (two workforce tiers). Should appear in distributions
        or relationships.
        """
        findings = _load_findings()
        sal_opaque   = planted["salary_opaque"]
        sal_semantic = planted["salary_semantic"]

        text = _findings_text(findings, "distributions", "relationships",
                              "subgroup_effects")
        assert _mentions(text, sal_opaque, sal_semantic), (
            f"Salary column (opaque: '{sal_opaque}', semantic: '{sal_semantic}') "
            f"not mentioned in findings. It is bimodal (two workforce tiers) and "
            f"predictive of churn — it must appear in findings."
        )

    def test_department_effect_identified(self, planted):
        """
        P3: churn rate varies significantly by department.
        Must appear in subgroup_effects or relationships.
        """
        findings = _load_findings()
        dept_opaque   = planted["dept_opaque"]
        dept_semantic = planted["dept_semantic"]

        dept_entries = [
            e for cat in ("subgroup_effects", "relationships")
            for e in findings.get(cat, [])
            if isinstance(e, dict)
            and _entry_mentions_variable(e, dept_opaque, dept_semantic,
                                         "Engineering", "Sales", "Support", "HR",
                                         "department", "dept")
        ]
        assert dept_entries, (
            f"No finding mentions the department column or any department name "
            f"(opaque: '{dept_opaque}', semantic: '{dept_semantic}'). "
            f"Engineering has the lowest churn and Sales the highest — this effect "
            f"must appear in findings['subgroup_effects'] or ['relationships']."
        )


# ── E. Narrative ──────────────────────────────────────────────────────────────

class TestNarrative:
    def test_minimum_word_count(self):
        summary = _load_summary()
        wc = len(summary.split())
        assert wc >= 100, f"summary.txt must be at least 100 words, found {wc}"

    def test_mentions_target_variable(self, planted):
        summary = _load_summary().lower()
        tgt_opaque   = planted["target_opaque"].lower()
        tgt_semantic = planted["target_semantic"].lower()
        assert tgt_opaque in summary or tgt_semantic in summary or \
               _mentions(summary, "churn", "attrition", "turnover", "left", "resign"), (
            f"summary.txt must mention the target variable ('{tgt_opaque}' or "
            f"'{tgt_semantic}') or use churn/attrition language."
        )

    def test_mentions_key_predictor(self, planted):
        summary = _load_summary().lower()
        sat_opaque   = planted["key_predictor_opaque"].lower()
        sat_semantic = planted["key_predictor_semantic"].lower()
        assert sat_opaque in summary or sat_semantic in summary or \
               _mentions(summary, "satisfaction", "satisfaction score",
                         "morale", "engagement"), (
            f"summary.txt should mention the key predictor "
            f"('{sat_opaque}' or '{sat_semantic}')."
        )

    def test_analytical_depth(self):
        summary = _load_summary().lower()
        keywords = [
            "churn", "predict", "driver", "factor", "correlation",
            "department", "salary", "satisfaction", "bimodal",
            "engineering", "sales", "subgroup", "analysis",
        ]
        hits = [kw for kw in keywords if kw in summary]
        assert len(hits) >= 4, (
            f"summary.txt mentions only {len(hits)} analytical keywords ({hits}); "
            f"expected at least 4. Write a substantive narrative."
        )
