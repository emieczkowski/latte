from __future__ import annotations
import asyncio
import csv
import json
import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
import matplotlib
from run_no_graph_pipeline import run_no_graph_pipeline, run_decentralized_pipeline
from run_metagpt_pipeline import run_metagpt_trial

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_dynamic_pipeline import run_pipeline

TASK_FILE = Path("tasks/simple_eda/task_description.json")

_PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-5.2"
}

CSV_FIELDS = [
    "condition", "trial", "n_teammates",
    "success", "tasks_done", "total_tasks", "task_completion_rate",
    "wall_clock_seconds", "total_rounds",
    "total_input_tokens", "total_output_tokens", "total_tokens",
    "tokens_per_task",
    "dispatched_agent_rounds", "possible_agent_rounds",
    "idle_agent_rounds", "idle_fraction",
    "run_dir",
]

def extract_event_metrics(run_dir: Path) -> dict:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return {
            "total_rounds": 0,
            "dispatched_agent_rounds": 0,
            "possible_agent_rounds": 0,
            "idle_agent_rounds": 0,
            "idle_fraction": 0.0,
        }
    events = [json.loads(line) for line in path.open()]
    dispatches   = [e for e in events if e["type"] == "round_dispatch"]
    total_rounds = sum(1 for e in events if e["type"] == "round_start")
    dispatched   = sum(d["n_dispatched"] for d in dispatches)
    possible     = sum(d["n_total"]      for d in dispatches)
    idle         = possible - dispatched
    return {
        "total_rounds":            total_rounds,
        "dispatched_agent_rounds": dispatched,
        "possible_agent_rounds":   possible,
        "idle_agent_rounds":       idle,
        "idle_fraction":           idle / possible if possible > 0 else 0.0,
    }


_MAX_METAGPT_RETRIES = 1


def run_one_trial(
    condition: str,
    trial_idx: int,
    n: int,
    exp_dir: Path,
    model_config: dict,
    max_rounds: int,
    straggler_rounds: int,
    verify_threshold: float | None,
) -> Optional[dict]:
    provider = model_config["provider"]
    model = model_config["model"].replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = exp_dir / "runs" / f"{provider}_{model}_{trial_idx:02d}_{condition}_{ts}"

    print(f"\n{'─' * 62}")
    print(f"  Condition: {condition.upper()}   Trial: {trial_idx}   n_teammates: {n}")
    print(f"{'─' * 62}")

    try:
        if condition == "metagpt":
            result = None
            for attempt in range(_MAX_METAGPT_RETRIES):
                if attempt > 0:
                    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_dir = exp_dir / "runs" / f"{provider}_{model}_{trial_idx:02d}_{condition}_{ts2}"
                    print(f"\n  [metagpt] attempt {attempt + 1}/{_MAX_METAGPT_RETRIES} — retrying in new run dir")
                result = run_metagpt_trial(
                    task_file=TASK_FILE,
                    run_dir=run_dir,
                    model_config=model_config,
                )
                if result is not None and result["success"]:
                    break
            if result is None:
                return None
        elif condition == "no_graph":
            result = asyncio.run(run_no_graph_pipeline(
                task_file=TASK_FILE,
                n_teammates=n,
                max_rounds=max_rounds,
                run_dir=run_dir,
                model_config=model_config,
                verbose=True,
                allow_run_tests=True,
            ))
        elif condition == "decentralized":
            result = asyncio.run(run_decentralized_pipeline(
                task_file=TASK_FILE,
                n_agents=n + 1,  # match total agents: other conditions have 1 lead + n teammates
                max_rounds=max_rounds,
                run_dir=run_dir,
                model_config=model_config,
                verbose=True,
                allow_run_tests=True,
            ))
        else:
            result = asyncio.run(run_pipeline(
                task_file=TASK_FILE,
                n_teammates=n,
                max_rounds=max_rounds,
                run_dir=run_dir,
                model_config=model_config,
                verbose=True,
                straggler_rounds=straggler_rounds,
                verify_threshold=verify_threshold,
                dynamic_agents=(condition == "dynamic"),
                static_graph=(condition == "static"),
            ))
    except Exception:
        print("  !! Trial crashed:")
        traceback.print_exc()
        return None

    ev = extract_event_metrics(run_dir)
    if condition == "metagpt":
        ev["total_rounds"] = result.get("total_rounds", 0)
    total_tok  = result["total_input_tokens"] + result["total_output_tokens"]
    tasks_done = result["tasks_done"]
    total_task = result["total_tasks"]

    row = {
        "condition":            condition,
        "trial":                trial_idx,
        "n_teammates":          n,
        "success":              result["success"],
        "tasks_done":           tasks_done,
        "total_tasks":          total_task,
        "task_completion_rate": tasks_done / total_task if total_task > 0 else 0.0,
        "wall_clock_seconds":   result["wall_clock_seconds"],
        "total_input_tokens":   result["total_input_tokens"],
        "total_output_tokens":  result["total_output_tokens"],
        "total_tokens":         total_tok,
        "tokens_per_task":      total_tok / tasks_done if tasks_done > 0 else float("nan"),
        "run_dir":              result["run_dir"],
        **ev,
    }

    status = "SUCCESS" if result["success"] else f"INCOMPLETE ({tasks_done}/{total_task})"
    print(f"\n  Result:        {status}")
    print(f"  Wall clock:    {result['wall_clock_seconds']:.1f}s   rounds: {ev['total_rounds']}")
    print(f"  Tokens:        {total_tok:,}  ({result['total_input_tokens']:,} in / {result['total_output_tokens']:,} out)")
    print(f"  Idle fraction: {ev['idle_fraction']:.1%}  ({ev['idle_agent_rounds']} idle of {ev['possible_agent_rounds']} possible agent-rounds)")

    return row


def _col(rows: list[dict], condition: str, key: str) -> np.ndarray:
    vals = [r[key] for r in rows if r["condition"] == condition]
    arr  = np.array(vals, dtype=float)
    return arr[~np.isnan(arr)]


def _bar(ax, dyn: np.ndarray, fix: np.ndarray, sta: np.ndarray, ylabel: str, title: str, pct: bool = False):
    arrays = [dyn, fix, sta]
    means  = [a.mean() if len(a) else 0.0     for a in arrays]
    stds   = [a.std()  if len(a) > 1 else 0.0 for a in arrays]
    colors = ["#2196F3", "#FF9800", "#66BB6A"]
    labels = ["Dynamic", "Fixed-N", "Static"]

    bars = ax.bar([0, 1, 2], means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, width=0.5,
                  error_kw={"linewidth": 1.5, "ecolor": "#333333"})
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)

    offset = max(stds) * 0.1 if max(stds) > 0 else (max(means) * 0.02 if max(means) > 0 else 0.01)
    for bar, mean in zip(bars, means):
        label_txt = f"{mean:.1%}" if pct else (f"{mean:,.0f}" if mean > 100 else f"{mean:.2f}")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label_txt,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )


def plot_results(rows: list[dict], out_dir: Path, n: int, k: int, provider: str, model: str) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    fig.suptitle(
        f"Experiment 1 — Dynamic vs Fixed-N vs Static Dispatch\n"
        f"n={n} teammates · up to {k} trials per condition · {provider} / {model}\n"
        f"Task: Synthetic EDA",
        fontsize=11, fontweight="bold", y=0.99,
    )

    _bar(axes[0, 0],
         _col(rows, "dynamic", "total_tokens"),
         _col(rows, "fixed",   "total_tokens"),
         _col(rows, "static",  "total_tokens"),
         "Tokens", "Total Tokens  (input + output)")

    _bar(axes[0, 1],
         _col(rows, "dynamic", "tokens_per_task"),
         _col(rows, "fixed",   "tokens_per_task"),
         _col(rows, "static",  "tokens_per_task"),
         "Tokens / task", "Tokens per Completed Task")

    _bar(axes[1, 0],
         _col(rows, "dynamic", "wall_clock_seconds"),
         _col(rows, "fixed",   "wall_clock_seconds"),
         _col(rows, "static",  "wall_clock_seconds"),
         "Seconds", "Wall-Clock Latency (s)")

    _bar(axes[1, 1],
         _col(rows, "dynamic", "total_rounds"),
         _col(rows, "fixed",   "total_rounds"),
         _col(rows, "static",  "total_rounds"),
         "Rounds", "Total Rounds")

    _bar(axes[2, 0],
         _col(rows, "dynamic", "task_completion_rate"),
         _col(rows, "fixed",   "task_completion_rate"),
         _col(rows, "static",  "task_completion_rate"),
         "Fraction of tasks done", "Task Completion Rate", pct=True)

    def _succ(cond): return np.array([1.0 if r["success"] else 0.0 for r in rows if r["condition"] == cond])
    _bar(axes[2, 1], _succ("dynamic"), _succ("fixed"), _succ("static"),
         "Fraction of trials", "Success Rate", pct=True)

    dyn_idle = _col(rows, "dynamic", "idle_fraction")
    fix_idle = _col(rows, "fixed",   "idle_fraction")
    sta_idle = _col(rows, "static",  "idle_fraction")
    parts = []
    if len(dyn_idle): parts.append(f"Dynamic: {dyn_idle.mean():.1%} ± {dyn_idle.std():.1%}")
    if len(fix_idle): parts.append(f"Fixed: {fix_idle.mean():.1%} ± {fix_idle.std():.1%}")
    if len(sta_idle): parts.append(f"Static: {sta_idle.mean():.1%} ± {sta_idle.std():.1%}")
    if parts:
        fig.text(0.5, 0.005, "Idle agent-rounds  —  " + "   |   ".join(parts),
                 ha="center", fontsize=9, color="#555555")

    for ax, label in zip(axes[:, 0], ["TOKEN EFFICIENCY", "LATENCY", "PERFORMANCE"]):
        ax.annotate(
            label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 30, 0),
            xycoords=ax.yaxis.label, textcoords="offset points",
            ha="right", va="center", fontsize=9, color="#777777",
            fontweight="bold", rotation=90,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = out_dir / "experiment1_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary(rows: list[dict]):
    metrics = [
        ("total_tokens",          "Total tokens"),
        ("tokens_per_task",       "Tokens / task"),
        ("idle_fraction",         "Idle agent-round frac."),
        ("wall_clock_seconds",    "Wall clock (s)"),
        ("total_rounds",          "Rounds"),
        ("task_completion_rate",  "Completion rate"),
    ]
    conditions = ["dynamic", "fixed", "static"]

    print(f"\n{'═' * 66}")
    print(f"  EXPERIMENT 1 RESULTS")
    print(f"{'═' * 66}")
    print(f"  {'Metric':<26}" + "".join(f"  {c:<18}" for c in conditions))
    print(f"  {'─' * 62}")
    for key, label in metrics:
        row_str = f"  {label:<26}"
        for c in conditions:
            vals = _col(rows, c, key)
            if len(vals):
                row_str += f"  {vals.mean():>8.3f} ±{vals.std():>7.3f}"
            else:
                row_str += f"  {'N/A':>16}"
        print(row_str)

    print(f"\n  Trials completed:")
    for c in conditions:
        n_trials = sum(1 for r in rows if r["condition"] == c)
        n_succ   = sum(1 for r in rows if r["condition"] == c and r["success"])
        print(f"    {c:<10}: {n_trials} trials, {n_succ} succeeded")
    print(f"{'═' * 66}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Dynamic vs Fixed-N Dispatch")
    parser.add_argument("--k",                type=int,   default=3,
                        help="Trials per condition (default: 3)")
    parser.add_argument("--n",                type=int,   default=4,
                        help="Number of teammate agents (default: 4)")
    parser.add_argument("--max-rounds",       type=int,   default=40,
                        help="Max rounds per trial (default: 40)")
    parser.add_argument("--straggler-rounds",   type=int,   default=None,
                        help="Auto-release straggler tasks after N rounds (default: disabled)")
    parser.add_argument("--provider",         type=str,   default="anthropic")
    parser.add_argument("--model",            type=str,   default=None)
    parser.add_argument(
        "--conditions",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated conditions to run (default: static,dynamic,no_graph,decentralized). "
             "Available: static, dynamic, no_graph, decentralized, metagpt",
    )
    args = parser.parse_args()

    model         = args.model or _PROVIDER_DEFAULTS.get(args.provider, args.provider)
    model_config  = {"provider": args.provider, "model": model}

    exp_dir = Path("experiments") / "experiment1" / f"n{args.n}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path    = exp_dir / "results.csv"
    csv_exists  = csv_path.exists()
    start_trial = 1
    if csv_exists:
        with csv_path.open() as f:
            existing = list(csv.DictReader(f))
        if existing:
            start_trial = max(int(r["trial"]) for r in existing) + 1

    print(f"\n{'═' * 62}")
    print(f"  EXPERIMENT 1 — Dynamic vs Fixed-N Dispatch")
    print(f"{'═' * 62}")
    print(f"  conditions      : dynamic, fixed, static")
    print(f"  n_teammates     : {args.n}")
    print(f"  trials (k)      : {args.k} per condition  ({args.k * 3} total)")
    print(f"  starting trial  : {start_trial}")
    print(f"  max_rounds      : {args.max_rounds}")
    print(f"  straggler_rounds: {args.straggler_rounds}")
    print(f"  verify_threshold: {args.verify_threshold}")
    print(f"  provider/model  : {args.provider} / {model}")
    print(f"  output dir      : {exp_dir}")
    print(f"{'═' * 62}")

    rows: list[dict] = []

    csv_mode = "a" if csv_exists else "w"
    with csv_path.open(csv_mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not csv_exists:
            writer.writeheader()

        active_conditions = args.conditions or ["dynamic", "no_graph", "decentralized", "static", "metagpt"]

        # Interleave conditions each trial to reduce time-of-day confounds
        for trial in range(start_trial, start_trial + args.k):

            for condition in active_conditions:
                row = run_one_trial(
                    condition=condition,
                    trial_idx=trial,
                    n=args.n,
                    exp_dir=exp_dir,
                    model_config=model_config,
                    max_rounds=args.max_rounds,
                    straggler_rounds=args.straggler_rounds,
                    verify_threshold=args.verify_threshold,
                )
                if row is None:
                    continue
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()  

    if not rows:
        print("\nNo trials completed successfully. Check errors above.")
        return 1

    print(f"\n  Results CSV → {csv_path}")

    chart_path = plot_results(rows, exp_dir, args.n, args.k, args.provider, model)
    print(f"  Chart       → {chart_path}")

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
