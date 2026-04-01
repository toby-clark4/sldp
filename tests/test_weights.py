from __future__ import annotations

import numpy as np

from sldp.weights import invert_weights


class TestInvertWeights:
    def test_invert_weights_identity_mode_returns_input(self) -> None:
        x = np.array([1.0, np.nan, -2.0])

        result = invert_weights(
            None, None, sigma2g=0.1, N=100.0, x=x, mode="Winv_ahat_I"
        )

        np.testing.assert_array_equal(result, x)

    def test_invert_weights_uses_typed_mask_for_heuristic_mode(self) -> None:
        x = np.array([4.0, np.nan, 8.0])
        typed = np.array([True, False, True])
        r = {"U": np.eye(3), "svs": np.array([2.0, 3.0, 4.0])}

        result = invert_weights(
            r, None, sigma2g=0.5, N=10.0, x=x, typed=typed, mode="Winv_ahat_h"
        )

        u = r["U"][typed, :]
        expected = np.full(x.shape, np.nan)
        expected[typed] = (u / (0.5 * r["svs"] ** 2 + r["svs"] / 10.0)).dot(
            u.T.dot(x[typed])
        )
        np.testing.assert_allclose(result[typed], expected[typed])
        assert np.isnan(result[1])
