import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SlopRankLogger")

@dataclass
class VisualizationConfig:
    """Configuration for graph visualization options."""
    enabled: bool = True
    save_formats: List[str] = field(default_factory=lambda: ["png", "html", "gml"])
    node_size_factor: float = 2000
    edge_width_factor: float = 2.0
    layout: str = "spring"  # Options: spring, circular, kamada_kawai, spectral
    node_colormap: str = "viridis"
    edge_colormap: str = "plasma"
    interactive: bool = True

@dataclass
class ConfidenceConfig:
    """Configuration for confidence interval calculations."""
    enabled: bool = True
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95  # e.g., 0.95 for 95% confidence interval
    significance_threshold: float = 0.05  # p-value threshold for significance

@dataclass
class WebDashboardConfig:
    """Configuration for the web dashboard."""
    enabled: bool = False  # Default to disabled
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    auto_open_browser: bool = True

@dataclass
class EvalConfig:
    """Configuration for the SlopRank evaluation system."""
    # Core configuration
    model_names: List[str]
    evaluation_method: int  # 1 => numeric rating, 2 => up/down (example usage)
    use_subset_evaluation: bool
    evaluators_subset_size: int
    output_dir: Path
    request_delay: float = 0.0
    
    # New features
    prompt_categories: Dict[str, List[str]] = field(default_factory=dict)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    web_dashboard: WebDashboardConfig = field(default_factory=WebDashboardConfig)
    
    # Optional metadata fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.evaluation_method not in {1, 2}:
            raise ValueError("evaluation_method must be 1 or 2")
        if self.evaluators_subset_size >= len(self.model_names):
            raise ValueError("evaluators_subset_size must be < number of models")
        
        # Create visualization directory if needed
        if self.visualization.enabled:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

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
    request_delay=0.0,
    # Default prompt categories (empty)
    prompt_categories={},
    # Default visualization configuration
    visualization=VisualizationConfig(
        enabled=True,
        save_formats=["png", "html", "gml"],
        node_size_factor=2000,
        edge_width_factor=2.0,
        layout="spring",
        node_colormap="viridis",
        edge_colormap="plasma",
        interactive=True
    ),
    # Default confidence configuration
    confidence=ConfidenceConfig(
        enabled=True,
        bootstrap_iterations=1000,
        confidence_level=0.95,
        significance_threshold=0.05
    ),
    # Default web dashboard configuration (disabled by default)
    web_dashboard=WebDashboardConfig(
        enabled=False,
        host="127.0.0.1",
        port=8050,
        debug=False,
        auto_open_browser=True
    )
)
