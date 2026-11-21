"""Seeded synthetic data generator used by demos and tests only."""

import numpy as np
import pandas as pd

FEATURES = ["avg_delivery_delay", "avg_carrier_handoff", "freight_ratio", "total_gmv"]


def make_synthetic_batch(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "avg_delivery_delay": rng.normal(0.5, 1.0, n),
            "avg_carrier_handoff": rng.normal(0.0, 1.0, n),
            "freight_ratio": rng.normal(0.0, 1.0, n),
            "total_gmv": rng.normal(0.0, 1.0, n),
        }
    )
    logit = (
        0.7 * frame.avg_delivery_delay
        + 0.6 * frame.avg_carrier_handoff
        - 0.45 * frame.freight_ratio
        + 0.2 * frame.total_gmv
        - 0.2
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame["_label_uniform"] = rng.random(n)
    frame["is_at_risk"] = (frame["_label_uniform"] < probability).astype(int)
    return frame
