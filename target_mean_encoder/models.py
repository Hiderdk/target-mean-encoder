from typing import Dict

from dataclasses import dataclass


@dataclass
class TargetMean:
    cell_name: str
    mean: float
    base_weight: float = None,
    weighted_mean: float = None
    observations: float = None
    sample_size_mean: float = None


@dataclass
class FeatureTargetMean:
    name: str
    target_means: Dict[str, TargetMean]
    if_unknown_mean: float = None

