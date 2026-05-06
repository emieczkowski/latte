"""
Called by the orchestrator before agents start so that employee_data.csv
exists in repo_dir at round 0.
"""
from pathlib import Path
from conftest import _build_dataset


def setup(repo_dir: Path):
    repo_dir = Path(repo_dir)
    df = _build_dataset(seed=42)
    df.to_csv(repo_dir / "employee_data.csv", index=False)
    (repo_dir / "figures").mkdir(exist_ok=True)
