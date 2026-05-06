"""
Run the private evaluation suite (test_suite_eval.py) against every run in
an experiment directory and write results to hidden_eval_results.csv.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

TASK_CONFIGS = {
    "simple_eda": {
        "task_dir":       PROJECT_ROOT / "tasks" / "simple_eda",
        "skip_if_missing": ["findings.json", "config.py", "summary.txt"],
        "test_classes": [
            "TestOutputContract",
            "TestConfigCompleteness",
            "TestConfigAccuracy",
            "TestPlantedFindings",
            "TestNarrative",
        ],
        "default_exp_dir": "experiments/experiment1/n4",
    },
    "textproc": {
        "task_dir":       PROJECT_ROOT / "tasks" / "textproc",
        "skip_if_missing": ["sentiment.py", "keywords.py", "pipeline.py"],
        "test_classes": [
            "TestDocumentExtensionsEval",
            "TestTokenizerExtensionsEval",
            "TestSentimentEval",
            "TestKeywordsEval",
            "TestSummarizerEval",
            "TestSimilarityEval",
            "TestFormatterEval",
            "TestPipelineIntegration",
        ],
        "default_exp_dir": "experiments/textproc/n4",
    },
}

BASE_FIELDNAMES = ["condition", "trial", "model", "run_dir",
                   "hidden_passed", "hidden_failed", "hidden_total", "hidden_pass_rate"]


def run_eval(repo_dir: Path, task_dir: Path, test_classes: list[str]) -> dict:
    """Run test_suite_eval.py in repo_dir and return pass/fail counts per class."""
    eval_suite = task_dir / "test_suite_eval.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(task_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(eval_suite), "-v", "--tb=no", "-q",
         "--no-header", "--override-ini=addopts="],
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    passed = failed = 0
    class_results = {c: None for c in test_classes}

    for line in output.splitlines():
        for cls in test_classes:
            if f"::{cls}::" in line:
                if " PASSED" in line:
                    class_results[cls] = (class_results[cls] or 0) + 1
                    passed += 1
                elif " FAILED" in line or " ERROR" in line:
                    if class_results[cls] is None:
                        class_results[cls] = 0
                    failed += 1

    # Fallback: parse summary line when verbose output is suppressed
    if passed == 0 and failed == 0:
        for line in output.splitlines():
            line = line.strip()
            if "passed" in line or "failed" in line or "error" in line:
                m_p = re.search(r"(\d+) passed", line)
                m_f = re.search(r"(\d+) failed", line)
                m_e = re.search(r"(\d+) error", line)
                if m_p:
                    passed = int(m_p.group(1))
                if m_f:
                    failed = int(m_f.group(1))
                if m_e:
                    failed += int(m_e.group(1))

    total = passed + failed
    rate  = passed / total if total > 0 else None
    return {
        "hidden_passed": passed,
        "hidden_failed": failed,
        "hidden_total":  total,
        "hidden_pass_rate": rate,
        **{f"class_{c}": class_results[c] for c in test_classes},
    }


def parse_run_dir_name(name: str):
    """Extract model string from a run dir name."""
    m = re.match(r"^(.+?)_(\d+)_(dynamic|fixed|no_graph|decentralized|static|metagpt).*$", name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return name, None, "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_CONFIGS), default="simple_eda",
                        help="Which task's hidden eval suite to run")
    parser.add_argument("--experiment-dir", default=None,
                        help="Path to experiment dir (contains results.csv). "
                             "Defaults to the task's canonical experiment dir.")
    args = parser.parse_args()

    cfg        = TASK_CONFIGS[args.task]
    task_dir   = cfg["task_dir"]
    test_classes = cfg["test_classes"]
    skip_files = cfg["skip_if_missing"]
    exp_dir    = PROJECT_ROOT / (args.experiment_dir or cfg["default_exp_dir"])

    results_csv = exp_dir / "results.csv"
    output_csv  = exp_dir / "hidden_eval_results.csv"
    fieldnames  = BASE_FIELDNAMES + [f"class_{c}" for c in test_classes]

    if not results_csv.exists():
        print(f"No results.csv at {results_csv}")
        sys.exit(1)

    with open(results_csv) as f:
        runs = list(csv.DictReader(f))

    print(f"Task:           {args.task}")
    print(f"Eval suite:     {task_dir / 'test_suite_eval.py'}")
    print(f"Experiment dir: {exp_dir}")
    print(f"Runs to eval:   {len(runs)}")
    print()

    rows = []
    for i, run in enumerate(runs):
        run_dir  = PROJECT_ROOT / run["run_dir"]
        repo_dir = run_dir / "repo"
        label    = f"[{i+1}/{len(runs)}] {run['condition']} trial {run['trial']} — {run_dir.name}"
        print(label)

        # MetaGPT writes outputs directly to run_dir (no repo/ subdir)
        if not repo_dir.exists():
            if run_dir.exists():
                repo_dir = run_dir
            else:
                print(f"  SKIP: repo dir not found")
                continue

        missing = [f for f in skip_files if not (repo_dir / f).exists()]
        if missing:
            print(f"  SKIP: missing output file(s): {missing}")
            rows.append({
                "condition": run["condition"],
                "trial":     run["trial"],
                "model":     run.get("model", parse_run_dir_name(run_dir.name)[0]),
                "run_dir":   run["run_dir"],
                "hidden_passed": 0,
                "hidden_failed": None,
                "hidden_total":  None,
                "hidden_pass_rate": None,
                **{f"class_{c}": None for c in test_classes},
            })
            continue

        metrics = run_eval(repo_dir, task_dir, test_classes)
        model   = run.get("model", parse_run_dir_name(run_dir.name)[0])
        rows.append({"condition": run["condition"], "trial": run["trial"],
                     "model": model, "run_dir": run["run_dir"], **metrics})

        if metrics["hidden_total"]:
            print(f"  → {metrics['hidden_passed']}/{metrics['hidden_total']} passed "
                  f"({metrics['hidden_pass_rate']:.0%})")
        else:
            print("  → no test results parsed")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {output_csv}")


if __name__ == "__main__":
    main()
