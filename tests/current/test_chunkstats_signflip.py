from __future__ import annotations

import numpy as np

from sldp.core.chunkstats import signflip


class TestSignflip:
    def test_signflip_sum_mode_returns_p_and_z(self) -> None:
        rng = np.random.default_rng(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5]), T=100000, printmem=False, mode="sum", rng=rng)

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_medrank_mode_returns_p_and_z(self) -> None:
        rng = np.random.default_rng(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5]), T=100000, printmem=False, mode="medrank", rng=rng)

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_thresh_mode_returns_p_and_z(self) -> None:
        rng = np.random.default_rng(123)

        p_value, z_value = signflip(np.array([1.0, -2.0, 0.5, 3.0]), T=100000, printmem=False, mode="thresh", rng=rng)

        assert 0 < p_value <= 1
        assert np.isfinite(z_value)

    def test_signflip_is_reproducible_with_same_generator_seed(self) -> None:
        q = np.array([1.0, -2.0, 0.5])

        first = signflip(q, T=100000, printmem=False, mode="sum", rng=np.random.default_rng(123))
        second = signflip(q, T=100000, printmem=False, mode="sum", rng=np.random.default_rng(123))

        assert first == second

    def test_signflip_does_not_mutate_global_numpy_random_state_when_rng_is_provided(self) -> None:
        q = np.array([1.0, -2.0, 0.5])
        np.random.seed(123)
        state_before = np.random.get_state()

        signflip(q, T=100000, printmem=False, mode="sum", rng=np.random.default_rng(456))

        state_after = np.random.get_state()
        assert state_before[0] == state_after[0]
        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2:] == state_after[2:]

    def test_signflip_invalid_mode_returns_none(self) -> None:
        assert signflip(np.array([1.0, 2.0]), T=10, printmem=False, mode="bad") is None
