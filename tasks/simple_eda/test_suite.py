"""
Public test suite — copied into the agent repo and available via <run_tests />.

Checks the OUTPUT CONTRACT only:
  - config.py exists, is importable, and has the required attributes
  - findings.json exists and has valid structure
  - summary.txt exists and meets minimum length

Does NOT check whether the column mapping is correct or whether planted
properties were discovered — those are in the private eval suite.
"""
import os
import sys
import json
import importlib.util
import pytest


def _cwd_path(*parts):
    return os.path.join(os.getcwd(), *parts)


def _load_config():
    path = _cwd_path("config.py")
    if not os.path.exists(path):
        pytest.fail("config.py not found in the repo root.")
    spec = importlib.util.spec_from_file_location("config", path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.fail(f"config.py raised an error on import: {e}")
    return mod


def _load_findings():
    path = _cwd_path("findings.json")
    if not os.path.exists(path):
        pytest.fail("findings.json not found in the repo root.")
    with open(path) as f:
        return json.load(f)


# ── A. config.py ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_exists(self):
        assert os.path.exists(_cwd_path("config.py")), \
            "config.py must be saved to the repo root"

    def test_config_importable(self):
        _load_config()

    def test_column_map_present(self):
        cfg = _load_config()
        assert hasattr(cfg, "COLUMN_MAP"), \
            "config.py must define COLUMN_MAP = {opaque_name: semantic_name, ...}"
        assert isinstance(cfg.COLUMN_MAP, dict), \
            "COLUMN_MAP must be a dict"
        assert len(cfg.COLUMN_MAP) > 0, \
            "COLUMN_MAP must not be empty"

    def test_target_present(self):
        cfg = _load_config()
        assert hasattr(cfg, "TARGET"), \
            "config.py must define TARGET = 'semantic_name_of_churn_column'"
        assert isinstance(cfg.TARGET, str) and cfg.TARGET.strip(), \
            "TARGET must be a non-empty string"

    def test_numeric_features_present(self):
        cfg = _load_config()
        assert hasattr(cfg, "NUMERIC_FEATURES"), \
            "config.py must define NUMERIC_FEATURES = [...]"
        assert isinstance(cfg.NUMERIC_FEATURES, list) and len(cfg.NUMERIC_FEATURES) > 0, \
            "NUMERIC_FEATURES must be a non-empty list"

    def test_categorical_features_present(self):
        cfg = _load_config()
        assert hasattr(cfg, "CATEGORICAL_FEATURES"), \
            "config.py must define CATEGORICAL_FEATURES = [...]"
        assert isinstance(cfg.CATEGORICAL_FEATURES, list) and len(cfg.CATEGORICAL_FEATURES) > 0, \
            "CATEGORICAL_FEATURES must be a non-empty list"

    def test_no_feature_overlap(self):
        cfg = _load_config()
        if not (hasattr(cfg, "NUMERIC_FEATURES") and hasattr(cfg, "CATEGORICAL_FEATURES")):
            pytest.skip("NUMERIC_FEATURES or CATEGORICAL_FEATURES not yet defined")
        overlap = set(cfg.NUMERIC_FEATURES) & set(cfg.CATEGORICAL_FEATURES)
        assert not overlap, \
            f"Columns appear in both NUMERIC_FEATURES and CATEGORICAL_FEATURES: {overlap}"


# ── B. findings.json ──────────────────────────────────────────────────────────

class TestFindingsJson:
    def test_exists(self):
        assert os.path.exists(_cwd_path("findings.json")), \
            "findings.json must be saved to the repo root"

    def test_valid_json(self):
        _load_findings()

    def test_required_keys(self):
        findings = _load_findings()
        required = {"distributions", "relationships", "subgroup_effects",
                    "outliers", "missing_data"}
        missing = required - set(findings.keys())
        assert not missing, f"findings.json missing top-level keys: {missing}"

    def test_all_categories_are_lists(self):
        findings = _load_findings()
        for key in ("distributions", "relationships", "subgroup_effects",
                    "outliers", "missing_data"):
            assert isinstance(findings.get(key), list), \
                f"findings['{key}'] must be a list"

    def test_all_categories_nonempty(self):
        findings = _load_findings()
        empty = [k for k in ("distributions", "relationships", "subgroup_effects",
                              "outliers", "missing_data")
                 if len(findings.get(k, [])) == 0]
        assert not empty, f"These categories need at least 1 entry: {empty}"

    def test_entries_have_finding_field(self):
        findings = _load_findings()
        for cat, entries in findings.items():
            if not isinstance(entries, list):
                continue
            for i, entry in enumerate(entries):
                assert isinstance(entry, dict), \
                    f"findings['{cat}'][{i}] must be a dict"
                assert "finding" in entry, \
                    f"findings['{cat}'][{i}] must have a 'finding' key"

    def test_entries_have_supporting_evidence(self):
        findings = _load_findings()
        for cat, entries in findings.items():
            if not isinstance(entries, list):
                continue
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                assert "supporting_evidence" in entry, (
                    f"findings['{cat}'][{i}] must have 'supporting_evidence' "
                    f"(include statistics, counts, R², etc.)"
                )
                assert str(entry["supporting_evidence"]).strip(), \
                    f"findings['{cat}'][{i}]['supporting_evidence'] must not be empty"


# ── C. summary.txt ────────────────────────────────────────────────────────────

class TestSummary:
    def test_exists(self):
        assert os.path.exists(_cwd_path("summary.txt")), \
            "summary.txt must be saved to the repo root"

    def test_minimum_length(self):
        text = open(_cwd_path("summary.txt")).read()
        words = len(text.split())
        assert words >= 100, \
            f"summary.txt must be at least 100 words, found {words}"

    def test_not_placeholder(self):
        text = open(_cwd_path("summary.txt")).read().strip().lower()
        assert text not in ("todo", "placeholder", "summary", ""), \
            "summary.txt appears to be a placeholder — write the actual narrative"
