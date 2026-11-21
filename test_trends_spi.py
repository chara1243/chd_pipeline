"""Unit tests for trend statistics and SPI."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.trends import (fdr_bh, mann_kendall, sens_slope,
                         trend_with_prewhitening, yue_pilon_prewhiten)


def test_mk_positive_trend():
    rng = np.random.default_rng(0)
    x = np.arange(50) * 0.4 + rng.normal(0, 0.5, 50)
    r = mann_kendall(x)
    assert r["z"] > 5
    assert r["p"] < 0.001


def test_mk_no_trend():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50)
    r = mann_kendall(x)
    assert r["p"] > 0.05


def test_sen_slope_recovers_true_slope():
    rng = np.random.default_rng(2)
    x = 2.0 * np.arange(40) + rng.normal(0, 1.0, 40)
    r = sens_slope(x)
    assert abs(r["slope"] - 2.0) < 0.3
    assert r["ci_lo"] < 2.0 < r["ci_hi"]


def test_prewhitening_runs_and_is_finite():
    # AR(1) noise with no trend: prewhitening must return a finite series and
    # a finite test result (sign of p-shift is statistical, not deterministic)
    rng = np.random.default_rng(3)
    x = np.zeros(200)
    for i in range(1, 200):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)
    r = trend_with_prewhitening(x, prewhiten=True)
    assert np.isfinite(r["p"]) and np.isfinite(r["slope"])


def test_fdr_bh_known_case():
    # 100 cells, 10 truly small p-values, rest uniform
    rng = np.random.default_rng(4)
    p = np.concatenate([rng.uniform(0, 1e-3, 10), rng.uniform(0.3, 0.9, 90)])
    mask = fdr_bh(p)
    assert mask.sum() >= 5
    assert fdr_bh(np.full(10, 0.9)).sum() == 0


def test_spi_marginals():
    sys.path.insert(0, str(ROOT))
    from core.spi import spi_series
    rng = np.random.default_rng(5)
    dates = np.arange(np.datetime64("1991-01-01"),
                      np.datetime64("2021-01-01"))
    # smooth 30-day-ish values: cumulative gamma noise
    v = rng.gamma(2.0, 3.0, len(dates))
    s = spi_series(v, dates)
    fin = s[np.isfinite(s)]
    assert fin.size > len(dates) * 0.5
    assert abs(np.mean(fin)) < 0.5          # standardised marginals
    assert abs(np.std(fin) - 1.0) < 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
