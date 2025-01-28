import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SlopRankLogger")

@dataclass
class EvalConfig:
    """Configuration for the SlopRank evaluation system."""
    model_names: List[str]
    evaluation_method: int  # 1 => numeric rating, 2 => up/down (example usage)
    use_subset_evaluation: bool
    evaluators_subset_size: int
    output_dir: Path
    request_delay: float = 0.0

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.evaluation_method not in {1, 2}:
            raise ValueError("evaluation_method must be 1 or 2")
        if self.evaluators_subset_size >= len(self.model_names):
            raise ValueError("evaluators_subset_size must be < number of models")

DEFAULT_CONFIG = EvalConfig(
    model_names=[
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-exp-1206",
        "claude-3-5-sonnet-latest",
        "o1-preview",
        "gpt-4o",
        "deepseek-chat"
    ],
    evaluation_method=1,  # numeric
    use_subset_evaluation=True,
    evaluators_subset_size=3,
    output_dir=Path("results"),
    request_delay=0.0
)
