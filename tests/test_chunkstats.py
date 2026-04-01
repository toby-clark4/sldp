from __future__ import annotations

import numpy as np
import pandas as pd

from sldp import chunkstats


class TestGetEst:
    def test_get_est_returns_solution_for_target_effect(self) -> None:
        num = np.array([2.0, 6.0])
        denom = np.array([[2.0, 0.0], [0.0, 3.0]])

        result = chunkstats.get_est(num, denom, k=0, num_background=1)

        assert result == 2.0


class TestJackknifeSe:
    def test_jackknife_se_returns_zero_for_identical_leave_one_out_estimates(
        self,
    ) -> None:
        loo_nums = [np.array([2.0]), np.array([2.0]), np.array([2.0])]
        loo_denoms = [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]

        result = chunkstats.jackknife_se(
            2.0, loo_nums, loo_denoms, k=0, num_background=0
        )

        assert result == 0.0


class TestResidualize:
    def test_residualize_without_background_returns_raw_terms(self) -> None:
        chunk_nums = [np.array([1.0]), np.array([3.0])]
        chunk_denoms = [np.array([[2.0]]), np.array([[4.0]])]

        q, r, mux, muy = chunkstats.residualize(
            chunk_nums, chunk_denoms, num_background=0, k=0
        )

        np.testing.assert_array_equal(q, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(r, np.array([2.0, 4.0]))
        np.testing.assert_array_equal(mux, np.array([]))
        np.testing.assert_array_equal(muy, np.array([]))


class TestCollapseToChunks:
    def test_collapse_to_chunks_aggregates_available_ldblocks(self) -> None:
        ldblocks = pd.DataFrame(
            {
                "chr": ["chr1", "chr1", "chr1"],
                "start": [0, 10, 20],
                "end": [10, 20, 30],
                "M_H": [2, 0, 3],
                "snpind_begin": [0, np.nan, 5],
                "snpind_end": [2, np.nan, 8],
            }
        )
        numerators = {0: np.array([1.0]), 2: np.array([4.0])}
        denominators = {0: np.array([[2.0]]), 2: np.array([[5.0]])}

        chunk_nums, chunk_denoms, loo_nums, loo_denoms, chunkinfo = (
            chunkstats.collapse_to_chunks(
                ldblocks,
                numerators,
                denominators,
                numblocks=2,
            )
        )

        assert len(chunk_nums) == 2
        np.testing.assert_array_equal(chunk_nums[0], np.array([1.0]))
        np.testing.assert_array_equal(chunk_nums[1], np.array([4.0]))
        np.testing.assert_array_equal(chunk_denoms[0], np.array([[2.0]]))
        np.testing.assert_array_equal(chunk_denoms[1], np.array([[5.0]]))
        assert len(loo_nums) == 2
        assert len(loo_denoms) == 2
        assert chunkinfo["numsnps"].tolist() == [2, 3]
