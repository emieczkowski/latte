import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 16000

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM config from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "8192"))
        )


@dataclass
class OrchestratorConfig:
    max_rounds: int = 20
    run_dir: Path = Path("runs/latest")
    repo_dir: Path = Path("repo")  # Relative to run_dir
    task_list_path: Path = Path("tasks/example_task.json")

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load orchestrator config from environment variables."""
        return cls(
            max_rounds=int(os.getenv("MAX_ROUNDS", "20")),
            run_dir=Path(os.getenv("RUN_DIR", "runs/latest")),
            repo_dir=Path(os.getenv("REPO_DIR", "repo")),
            task_list_path=Path(os.getenv("TASK_LIST_PATH", "tasks/example_task.json"))
        )


@dataclass
class MonitoringConfig:
    use_wandb: bool = False
    use_weave: bool = False
    wandb_project: str = "llm-team-design"
    weave_project: str = "llm-team-design"

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        wandb_key = os.getenv("WANDB_API_KEY")
        weave_project = os.getenv("WEAVE_PROJECT")

        return cls(
            use_wandb=bool(wandb_key),
            use_weave=bool(weave_project),
            wandb_project=os.getenv("WANDB_PROJECT", "llm-team-design"),
            weave_project=weave_project or "llm-team-design"
        )


@dataclass
class Config:
    llm: LLMConfig
    orchestrator: OrchestratorConfig
    monitoring: MonitoringConfig

    @classmethod
    def load(cls) -> "Config":
        """Load full configuration from environment."""
        return cls(
            llm=LLMConfig.from_env(),
            orchestrator=OrchestratorConfig.from_env(),
            monitoring=MonitoringConfig.from_env()
        )


def get_config() -> Config:
    return Config.load()
