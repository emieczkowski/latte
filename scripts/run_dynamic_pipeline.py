import asyncio
import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.lead_agent import LeadAgent
from agents.teammate_agent import TeammateAgent
from orchestrator.orchestrator import Orchestrator

MAX_ROUNDS = 40

_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-5.2"
}


def _load_prompts(static_graph=False, dynamic_agents=False):
    prompts_dir = Path("agents/prompts")
    lead_file   = "lead_prompt_static.txt" if static_graph else "lead_prompt.txt"
    lead_prompt     = (prompts_dir / lead_file).read_text()
    if static_graph:
        teammate_file = "teammate_prompt_static.txt"
    elif dynamic_agents:
        teammate_file = "dynamic_teammate_prompt.txt"
    else:
        teammate_file = "teammate_prompt.txt"
    teammate_prompt = (prompts_dir / teammate_file).read_text()
    planning_file   = "planning_prompt_static.txt" if static_graph else "planning_prompt.txt"
    planning_prompt = (prompts_dir / planning_file).read_text()
    return lead_prompt, teammate_prompt, planning_prompt


def create_team(num_teammates, model_config=None, static_graph=False, dynamic_agents=False, agents_override=None):
    """Build the lead + teammate agent dict.

    If *agents_override* is provided it is returned as-is, allowing callers to
    supply pre-built agents (e.g. with faulty variants injected) while still
    reusing all the rest of the pipeline setup.
    """
    if agents_override is not None:
        return agents_override
    lead_prompt, teammate_prompt, _ = _load_prompts(static_graph=static_graph, dynamic_agents=dynamic_agents)
    teammate_names = [f"Dev{i+1}" for i in range(num_teammates)]
    agents = {
        "Lead": LeadAgent(
            name="Lead",
            system_prompt=lead_prompt,
            team_members=teammate_names,
            model=model_config,
            static_graph=static_graph,
        )
    }
    for name in teammate_names:
        agents[name] = TeammateAgent(name=name, system_prompt=teammate_prompt, model=model_config)
    return agents


async def run_pipeline(task_file, n_teammates, max_rounds, run_dir, model_config=None, verbose=True, straggler_rounds=None, verify_threshold=None, dynamic_agents=False, static_graph=False, allow_run_tests=True, agents_override=None):
    task_file = Path(task_file)
    task_dir  = task_file.parent

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".locks").mkdir(exist_ok=True)
    repo_dir = run_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    task_copy = run_dir / "tasks.json"
    shutil.copy(task_file, task_copy)

    for fname in ("test_suite.py",):
        src = task_dir / fname
        if src.exists():
            shutil.copy(src, repo_dir / fname)

    setup_script = task_dir / "setup_data.py"
    if setup_script.exists():
        import importlib.util, sys as _sys
        _inserted = str(task_dir) not in _sys.path
        if _inserted:
            _sys.path.insert(0, str(task_dir))
        try:
            spec = importlib.util.spec_from_file_location("setup_data", setup_script)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "setup"):
                mod.setup(repo_dir)
        finally:
            if _inserted:
                _sys.path.remove(str(task_dir))

    _, _, planning_prompt = _load_prompts(static_graph=static_graph, dynamic_agents=dynamic_agents)
    agents = create_team(n_teammates, model_config=model_config, static_graph=static_graph,
                         dynamic_agents=dynamic_agents, agents_override=agents_override)

    orchestrator = Orchestrator(
        run_dir=run_dir,
        repo_dir=repo_dir,
        agents=agents,
        task_list_path=task_copy,
        straggler_rounds=straggler_rounds,
        static_graph=static_graph,
        dynamic_agents=dynamic_agents,
        allow_run_tests=allow_run_tests,
    )
    orchestrator.verify_exposure_threshold = verify_threshold

    if verbose:
        print("  [planning] Lead decomposing project into task graph...")
    n_tasks = await orchestrator.planning_phase(
        lead_agent_name="Lead",
        planning_prompt=planning_prompt,
    )

    agents["Lead"].history = []

    if verbose:
        tasks_data = json.loads(task_copy.read_text())
        for t in tasks_data["tasks"]:
            deps = f" <- {t['dependencies']}" if t["dependencies"] else ""
            print(f"    task {t['id']}: {t['title']}{deps}")
        print(f"  [planning] {n_tasks} tasks discovered.\n")

    orchestrator.initialize_agents(lead_agent_name="Lead")
    t0 = time.time()
    success = await orchestrator.run_async(max_rounds=max_rounds, lead_agent_name="Lead")
    wall_clock = time.time() - t0

    total_input, total_output = 0, 0
    for agent in agents.values():
        client = getattr(agent, "llm_client", None)
        if client:
            total_input  += client.total_input_tokens
            total_output += client.total_output_tokens

    tasks_data  = json.loads(task_copy.read_text())
    tasks_done  = sum(1 for t in tasks_data["tasks"] if t["status"] == "done")
    total_tasks = len(tasks_data["tasks"])

    return {
        "success": success,
        "tasks_done": tasks_done,
        "total_tasks": total_tasks,
        "wall_clock_seconds": wall_clock,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "run_dir": str(run_dir),
        "n_teammates": n_teammates,
        "task_file": str(task_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run any dynamic-graph task (NL description -> task graph -> implementation)"
    )
    parser.add_argument("--tasks",           type=str, required=True,      help="Path to task_description.json")
    parser.add_argument("--n",               type=int, default=3,          help="Number of teammate agents (default: 3)")
    parser.add_argument("--max-rounds",      type=int, default=MAX_ROUNDS, help=f"Max rounds for main loop (default: {MAX_ROUNDS})")
    parser.add_argument("--provider",        type=str, default=None,       help="LLM provider (anthropic/openai)")
    parser.add_argument("--model",           type=str, default=None,       help="Model name override")
    parser.add_argument("--straggler-rounds",  type=int,   default=None, help="Auto-release tasks held for N rounds without completing. Omit to disable (ablation baseline).")
    parser.add_argument("--dynamic-agents",    action="store_true",       help="Only dispatch agents that have work this round (frontier-capped). Default: fixed-N.")
    args = parser.parse_args()

    model_config = {}
    if args.provider:
        model_config["provider"] = args.provider
    if args.model:
        model_config["model"] = args.model
    model_config = model_config or None

    effective_provider = args.provider or os.getenv("LLM_PROVIDER", "anthropic")
    effective_model    = (args.model or os.getenv("LLM_MODEL")
                          or _PROVIDER_DEFAULT_MODELS.get(effective_provider, effective_provider))

    task_name = Path(args.tasks).parent.name 
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path("runs") / f"{effective_provider}_{effective_model}" / f"{task_name}_n{args.n}_{ts}"

    print(f"\nDynamic pipeline run")
    print(f"  task:      {args.tasks}")
    print(f"  provider:  {effective_provider}  model: {effective_model}")
    straggler_label = f"{args.straggler_rounds} rounds" if args.straggler_rounds else "disabled"
    verify_label    = f">= {args.verify_threshold}" if args.verify_threshold is not None else "disabled"
    print(f"  teammates: {args.n}  max_rounds: {args.max_rounds}")
    print(f"  straggler_release: {straggler_label}  verify_threshold: {verify_label}  dynamic_agents: {args.dynamic_agents}")
    print(f"  output:    {run_dir}\n")

    result = asyncio.run(run_pipeline(
        task_file=args.tasks,
        n_teammates=args.n,
        max_rounds=args.max_rounds,
        run_dir=run_dir,
        model_config=model_config,
        verbose=True,
        straggler_rounds=args.straggler_rounds,
        verify_threshold=args.verify_threshold,
        dynamic_agents=args.dynamic_agents,
    ))

    status = "SUCCESS" if result["success"] else f"INCOMPLETE ({result['tasks_done']}/{result['total_tasks']} tasks done)"
    print(f"Result: {status}")
    print(f"  Wall clock:    {result['wall_clock_seconds']:.1f}s")
    print(f"  Tasks done:    {result['tasks_done']} / {result['total_tasks']}")
    print(f"  Input tokens:  {result['total_input_tokens']:,}")
    print(f"  Output tokens: {result['total_output_tokens']:,}")
    print(f"  Run dir:       {result['run_dir']}")

    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
