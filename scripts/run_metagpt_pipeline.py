"""
run_metagpt_pipeline.py — run a MetaGPT team on a task from inside
the experiment dispatch infrastructure.

Drop-in alongside run_dynamic_pipeline / run_no_graph_pipeline:
  result = run_metagpt_trial(task_file, run_dir, model_config)

Returns the same dict shape as run_pipeline() so dispatch scripts
can treat MetaGPT as just another condition.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
VENV_PYTHON = REPO_ROOT / "vendor" / "venv-metagpt-paper" / "bin" / "python"
_CONFIG_SRC = REPO_ROOT / "metagpt_utils" / "metagpt_config.yaml"

# Maps task-dir name → (output_file, use_preloaded_qa)
# output_file may be comma-separated for tasks that produce multiple files.
_TASK_CONFIG: dict[str, tuple[str, bool]] = {
    "debug_sprint":  ("signal_toolkit.py",                      True),
    "bugfix_sprint": ("signal_toolkit.py",                      True),
    "search_lib":    ("search_lib.py",                          False),
    "simple_eda":    ("config.py,findings.json,summary.txt",    True),
    "textproc":      (
        "document.py,tokenizer.py,sentiment.py,keywords.py,"
        "summarizer.py,similarity.py,formatter.py,pipeline.py",
        False,
    ),
}

_N_TEAMMATES = 5   # PM, Architect, PM, Engineer, QaEngineer
_N_ROUND     = 30


def _load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def _setup_run_dir(run_dir: Path, task_file: Path, model_config: dict) -> None:
    """Populate run_dir with everything MetaGPT needs."""
    task_dir = task_file.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / ".project_root").touch()

    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    shutil.copy(_CONFIG_SRC, config_dir / "config.yaml")

    env    = _load_env()
    prov   = model_config.get("provider", "anthropic")
    if prov == "anthropic":
        key_yaml = f'Anthropic_API_KEY: "{env.get("ANTHROPIC_API_KEY", "")}"\n'
    elif prov == "openai":
        key_yaml = f'OPENAI_API_KEY: "{env.get("OPENAI_API_KEY", "")}"\n'
    else:
        key_yaml = ""
    (config_dir / "key.yaml").write_text(key_yaml)
    (run_dir / "key.yaml").write_text(key_yaml)

    # test_suite.py goes in the MetaGPT project root (run_dir)
    ts = task_dir / "test_suite.py"
    if ts.exists():
        shutil.copy(ts, run_dir / "test_suite.py")

    # conftest.py — needed so pytest fixtures (e.g. setup_dataset) run correctly
    cf = task_dir / "conftest.py"
    if cf.exists():
        shutil.copy(cf, run_dir / "conftest.py")

    # Run setup_data.py to plant buggy/stub source files
    setup_script = task_dir / "setup_data.py"
    if setup_script.exists():
        _inserted = str(task_dir) not in sys.path
        if _inserted:
            sys.path.insert(0, str(task_dir))
        try:
            spec = importlib.util.spec_from_file_location("setup_data", setup_script)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "setup"):
                mod.setup(run_dir)   # MetaGPT works in run_dir, not a repo/ subdir
        finally:
            if _inserted:
                sys.path.remove(str(task_dir))


def _write_launch_py(run_dir: Path, task_file: Path, model_config: dict) -> Path:
    """Generate a fresh launch.py from the shared LAUNCH_TEMPLATE."""
    task_name  = task_file.parent.name
    prov       = model_config.get("provider", "anthropic")
    model      = model_config.get("model", "claude-sonnet-4-6")
    idea       = json.loads(task_file.read_text()).get("description", "").strip()

    output_file, use_preloaded_qa = _TASK_CONFIG.get(
        task_name, ("output.py", False)
    )

    # For tasks with a data file, embed the absolute path so agents can find
    # employee_data.csv from inside the MetaGPT workspace subdirectory.
    if task_name == "simple_eda":
        idea = idea.replace("employee_data.csv", str(run_dir / "employee_data.csv"))

    # Import LAUNCH_TEMPLATE without executing setup_metagpt_runs module-level code
    with open(REPO_ROOT / "scripts" / "setup_metagpt_runs.py") as f:
        src = f.read()
    start = src.index("LAUNCH_TEMPLATE = '''")
    end   = src.index("'''", start + len("LAUNCH_TEMPLATE = '''")) + 3
    template = eval(src[start + len("LAUNCH_TEMPLATE = "):end])

    launch_src = template.format(
        venv_python      = str(VENV_PYTHON),
        repo_root        = str(REPO_ROOT),
        task             = task_name,
        provider_label   = f"{prov}/{model}",
        model            = model,
        llm_provider     = prov,
        idea             = idea,
        use_preloaded_qa = use_preloaded_qa,
        output_file      = output_file,
        # exp_dir/trial are unused when METAGPT_SKIP_CSV=1
        exp_dir          = "experiments/experiment2/n5",
        n_teammates      = _N_TEAMMATES,
        n_round          = _N_ROUND,
    )
    launch_path = run_dir / "launch.py"
    launch_path.write_text(launch_src)
    launch_path.chmod(0o755)
    return launch_path


def run_metagpt_trial(
    task_file,
    run_dir,
    model_config: dict,
    max_rounds: int | None = None,
) -> dict | None:
    """
    Run a MetaGPT team on task_file inside run_dir.

    Returns a dict matching run_pipeline()'s shape, or None on crash.
    """
    task_file = Path(task_file).resolve()
    run_dir   = Path(run_dir).resolve()

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(
            f"MetaGPT venv not found at {VENV_PYTHON}.\n"
            "Create it with:\n"
            "  python3 -m venv vendor/venv-metagpt-paper\n"
            "  vendor/venv-metagpt-paper/bin/pip install -e vendor/metagpt-paper\n"
            "  vendor/venv-metagpt-paper/bin/pip install anthropic openai python-dotenv"
        )

    _setup_run_dir(run_dir, task_file, model_config)
    launch_path = _write_launch_py(run_dir, task_file, model_config)

    env = os.environ.copy()
    env["METAGPT_SKIP_CSV"] = "1"   # dispatch script owns CSV writing

    proc = subprocess.run(
        [str(VENV_PYTHON), str(launch_path)],
        cwd=str(run_dir),
        env=env,
    )

    result_path = run_dir / "result.json"
    if not result_path.exists():
        print(f"  [metagpt] result.json missing — run likely crashed (rc={proc.returncode})")
        return None

    data = json.loads(result_path.read_text())
    # Normalise to the shape run_pipeline() returns
    return {
        "success":              data["success"],
        "tasks_done":           data["tasks_done"],
        "total_tasks":          data["total_tasks"],
        "wall_clock_seconds":   data["wall_clock_seconds"],
        "total_input_tokens":   data["total_input_tokens"],
        "total_output_tokens":  data["total_output_tokens"],
        "total_rounds":         data.get("total_rounds", data.get("rounds_run", 0)),
        "run_dir":              data["run_dir"],
        "n_teammates":          data.get("n_teammates", _N_TEAMMATES),
        "task_file":            str(task_file),
    }
