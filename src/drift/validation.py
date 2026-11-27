"""Binomial uncertainty for independent synthetic monitoring repetitions."""

from math import sqrt


def wilson_interval(successes: int, trials: int) -> list[float]:
    """95% Wilson interval for a count; never a CI for a designed six-window demo."""
    if (
        not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials <= 0
        or not 0 <= successes <= trials
    ):
        raise ValueError(
            "Expected integer counts with 0 <= successes <= positive trials."
        )
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]
