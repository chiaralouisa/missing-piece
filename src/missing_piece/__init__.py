"""missing-piece: reconstructing unassayed tumour genomic profiles from targeted panels."""

from .panels import Panel, PanelPair, load_panel_file, load_panels
from .data.cohort import Cohort, build_cohort
from .data.simulate import SimulationConfig, simulate_cohort
from .data.splits import SplitSpec, Splits, make_splits
from .experiment import ExperimentConfig, ExperimentResult, run_experiment
from .models import MODEL_REGISTRY, build_model

__version__ = "0.1.0"

__all__ = [
    "Panel",
    "PanelPair",
    "load_panel_file",
    "load_panels",
    "Cohort",
    "build_cohort",
    "SimulationConfig",
    "simulate_cohort",
    "SplitSpec",
    "Splits",
    "make_splits",
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
    "MODEL_REGISTRY",
    "build_model",
    "__version__",
]
