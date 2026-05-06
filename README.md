# LATTE
Official implementation of LATTE from "Improving the Efficiency of Language Agent Teams with Adaptive Task Graphs"

## Setup

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your API key(s):

```bash
# Use one or more providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Select provider and model (defaults to Anthropic / claude-sonnet-4-6)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
```

Supported providers: `anthropic`, `openai`.

## Running LATTE on a task

All commands are run from the repo root.

**Run LATTE (dynamic condition) on a single task:**
```bash
python scripts/run_dynamic_pipeline.py \
    --tasks tasks/simple_eda/task_description.json \
    --n 4
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | *(required)* | Path to `task_description.json` |
| `--n` | `3` | Number of teammate agents |
| `--max-rounds` | `40` | Round limit |
| `--provider` | env / `anthropic` | LLM provider |
| `--model` | env / provider default | Model name |
| `--straggler-rounds` | disabled | Automatically release tasks held for N rounds without progress. Otherwise, the lead will receive a heartbeat notification. |

Results are written to `runs/<provider>_<model>/<task>_n<N>_<timestamp>/`.

## Included tasks

| Task | Directory | Description |
|------|-----------|-------------|
| Simple EDA | `tasks/simple_eda/` | Analyze an opaque HR dataset; produce `config.py`, `findings.json`, `summary.txt` |
| Debug Sprint | `tasks/debug_sprint/` | Find and fix bugs in a signal-processing library (`signal_toolkit.py`) |
| Text Processing | `tasks/textproc/` | Extend an existing text processing library with new modules |

## Baselines

All four baseline conditions used in the paper are included:

| Condition | Description |
|-----------|-------------|
| `dynamic` | **LATTE** — adaptive task graph with straggler detection and verification |
| `static` | Task graph planned once upfront; no updates during execution |
| `no_graph` | No task graph; agents coordinate freely via broadcasts and file edits |
| `decentralized` | No lead agent; all agents self-coordinate without a graph |
| `metagpt` | MetaGPT multi-agent baseline |

To run a specific baseline directly:
```bash
# No-graph baseline
python scripts/run_no_graph_pipeline.py \
    --tasks tasks/debug_sprint/task_description.json \
    --n 4

# MetaGPT baseline
python scripts/run_metagpt_pipeline.py \
    --tasks tasks/textproc/task_description.json
```

## Reproducing paper experiments

Each dispatch script runs all conditions for one task and writes results to `experiments/`:

```bash
# Simple EDA (experiment 1)
python scripts/experiment1_dispatch.py --k 10 --n 4 --provider anthropic

# Debug Sprint (experiment 2)
python scripts/experiment2_dispatch.py --k 10 --n 4 --provider anthropic

# Text Processing (experiment 3)
python scripts/experiment_textproc_dispatch.py --k 10 --n 4 --provider anthropic
```

Key options for dispatch scripts:
| Flag | Default | Description |
|------|---------|-------------|
| `--k` | `3` | Trials per condition |
| `--n` | `4` | Number of teammate agents |
| `--conditions` | all | Comma-separated subset, e.g. `dynamic,static` |
| `--provider` / `--model` | env defaults | LLM provider and model |

Trials are interleaved across conditions to reduce time-of-day confounds. Results append to `experiments/<task>/n<N>/results.csv` so interrupted runs can be resumed.

## Analyzing results

Open `notebooks/run_visualizer_overview.ipynb` in Jupyter. The notebook loads all three `results.csv` files and produces the figures from the paper: pass rates, token and wall-clock costs, coordination metrics, LATTE operator usage, task-graph evolution.

Pre-computed experiment data (results CSVs and `events.jsonl` logs) is included in `experiments/`. 

## Adding a new task

1. Create `tasks/<your_task>/task_description.json` with `project`, `description`, `success_criteria`, and `tasks: []` fields.
2. Add `tasks/<your_task>/test_suite.py` with a pytest suite that defines success.
3. Optionally add `tasks/<your_task>/setup_data.py` with a `setup(repo_dir)` function to stage data files before agents start.
4. Run with `python scripts/run_dynamic_pipeline.py --tasks tasks/<your_task>/task_description.json`.

## License

MIT
