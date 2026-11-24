import pytest

from src.drift.validation import wilson_interval


def test_wilson_handles_zero_and_all_events_without_zero_width():
    low = wilson_interval(0, 100)
    high = wilson_interval(100, 100)
    assert low[0] == pytest.approx(0)
    assert 0.03 < low[1] < 0.04
    assert high[0] == pytest.approx(1 - low[1])
    assert high[1] == pytest.approx(1)


def test_larger_sample_narrows_interval():
    small, large = wilson_interval(5, 10), wilson_interval(50, 100)
    assert large[1] - large[0] < small[1] - small[0]


@pytest.mark.parametrize("count,total", [(-1, 10), (11, 10), (0, 0), (1.5, 10)])
def test_invalid_binomial_counts_are_rejected(count, total):
    with pytest.raises(ValueError):
        wilson_interval(count, total)
