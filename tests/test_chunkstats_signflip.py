from __future__ import annotations

import numpy as np

from sldp.chunkstats import signflip


class TestSignflip:
    def test_signflip_sum_mode_returns_p_and_z(self) -> None:
        np.random.seed(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5]), T=100000, printmem=False, mode="sum")

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_medrank_mode_returns_p_and_z(self) -> None:
        np.random.seed(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5]), T=100000, printmem=False, mode="medrank")

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_thresh_mode_returns_p_and_z(self) -> None:
        np.random.seed(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5, 3.0]), T=100000, printmem=False, mode="thresh")

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_invalid_mode_returns_none(self) -> None:
        assert signflip(np.array([1.0, 2.0]), T=10, printmem=False, mode="bad") is None
