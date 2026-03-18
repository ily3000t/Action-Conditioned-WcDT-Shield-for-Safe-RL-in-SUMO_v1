from typing import Iterable, Optional

import numpy as np


def aggregate_tail_risk(
    risk_values: Iterable[float],
    quantile: float = 0.9,
    uncertainty: Optional[float] = None,
    uncertainty_weight: float = 0.2,
) -> float:
    values = np.array(list(risk_values), dtype=np.float32)
    if values.size == 0:
        base = 1.0
    else:
        q = float(np.clip(quantile, 0.0, 1.0))
        base = float(np.quantile(values, q))
    if uncertainty is not None:
        base += float(np.clip(uncertainty, 0.0, 1.0)) * float(max(0.0, uncertainty_weight))
    return float(np.clip(base, 0.0, 1.0))
