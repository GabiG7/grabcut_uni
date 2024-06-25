import numpy as np
from dataclasses import dataclass


@dataclass
class GMM:
    mean: np.ndarray
    covariance_inverse: np.ndarray
    covariance_det: float
    component_weight: float
