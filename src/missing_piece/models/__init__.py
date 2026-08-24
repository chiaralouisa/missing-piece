"""Panel-completion models."""

from .base import FeatureSpec, ModelInputs, PanelCompletionModel
from .baselines import (
    BurdenBaseline,
    LogisticBaseline,
    MLPBaseline,
    PrevalenceBaseline,
)
from .flow import DiscreteFlowMatching, GaussianFlowMatching

#: Registry used by the config-driven runner.
MODEL_REGISTRY: dict[str, type[PanelCompletionModel]] = {
    "prevalence": PrevalenceBaseline,
    "burden": BurdenBaseline,
    "logistic": LogisticBaseline,
    "mlp": MLPBaseline,
    "flow_gaussian": GaussianFlowMatching,
    "flow_discrete": DiscreteFlowMatching,
}


def build_model(name: str, **kwargs) -> PanelCompletionModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "FeatureSpec",
    "ModelInputs",
    "PanelCompletionModel",
    "PrevalenceBaseline",
    "BurdenBaseline",
    "LogisticBaseline",
    "MLPBaseline",
    "GaussianFlowMatching",
    "DiscreteFlowMatching",
    "MODEL_REGISTRY",
    "build_model",
]
