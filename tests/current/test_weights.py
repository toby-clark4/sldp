from __future__ import annotations

import numpy as np

from sldp.core.weights import invert_weights


class TestInvertWeights:
    def test_invert_weights_identity_mode_returns_input(self) -> None:
        x = np.array([1.0, np.nan, -2.0])

        result = invert_weights(None, None, sigma2g=0.1, N=100.0, x=x, mode="Winv_ahat_I")

        np.testing.assert_array_equal(result, x)

    def test_invert_weights_uses_typed_mask_for_heuristic_mode(self) -> None:
        x = np.array([4.0, np.nan, 8.0])
        typed = np.array([True, False, True])
        r = {"U": np.eye(3), "svs": np.array([2.0, 3.0, 4.0])}

        result = invert_weights(r, None, sigma2g=0.5, N=10.0, x=x, typed=typed, mode="Winv_ahat_h")

        u = r["U"][typed, :]
        expected = np.full(x.shape, np.nan)
        expected[typed] = (u / (0.5 * r["svs"] ** 2 + r["svs"] / 10.0)).dot(u.T.dot(x[typed]))
        np.testing.assert_allclose(result[typed], expected[typed])
        assert np.isnan(result[1])

    def test_invert_weights_hln_mode_uses_large_n_approximation(self) -> None:
        x = np.array([4.0, np.nan, 8.0])
        typed = np.array([True, False, True])
        r = {"U": np.eye(3), "svs": np.array([2.0, 3.0, 4.0])}

        result = invert_weights(r, None, sigma2g=0.5, N=10.0, x=x, typed=typed, mode="Winv_ahat_hlN")

        u = r["U"][typed, :]
        expected = np.full(x.shape, np.nan)
        expected[typed] = (u / (r["svs"] ** 2)).dot(u.T.dot(x[typed]))
        np.testing.assert_allclose(result[typed], expected[typed])
        assert np.isnan(result[1])

    def test_invert_weights_h2_mode_uses_r2_spectrum(self) -> None:
        x = np.array([4.0, np.nan, 8.0])
        typed = np.array([True, False, True])
        r2 = {"U": np.eye(3), "svs": np.array([4.0, 9.0, 16.0])}

        result = invert_weights(None, r2, sigma2g=0.5, N=10.0, x=x, typed=typed, mode="Winv_ahat_h2")

        u = r2["U"][typed, :]
        expected = np.full(x.shape, np.nan)
        expected[typed] = (u / (0.5 * r2["svs"] + np.sqrt(r2["svs"]) / 10.0)).dot(u.T.dot(x[typed]))
        np.testing.assert_allclose(result[typed], expected[typed])
        assert np.isnan(result[1])

    def test_invert_weights_exact_mode_returns_low_rank_solution(self) -> None:
        x = np.array([5.0, np.nan, 7.0])
        typed = np.array([True, False, True])
        r = {"U": np.eye(3), "svs": np.array([2.0, 3.0, 4.0])}
        r2 = {"U": np.eye(3), "svs": np.array([1.0, 1.5, 2.0])}

        result = invert_weights(r, r2, sigma2g=0.5, N=10.0, x=x, typed=typed, mode="Winv_ahat")

        r_matrix = (r["U"][typed] * r["svs"]).dot(r["U"][typed].T)
        r2_matrix = (r2["U"][typed] * r2["svs"]).dot(r2["U"][typed].T)
        w = r_matrix / 10.0 + 0.5 * r2_matrix
        u, svs, _ = np.linalg.svd(w)
        k = np.argmax(np.cumsum(svs) / svs.sum() >= 0.95)
        u = u[:, :k]
        svs = svs[:k]
        expected = np.full(x.shape, np.nan)
        expected[typed] = (u / svs).dot(u.T.dot(x[typed]))

        np.testing.assert_allclose(result[typed], expected[typed])
        assert np.isnan(result[1])

    def test_invert_weights_infers_typed_mask_from_finite_rows(self) -> None:
        x = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        r = {"U": np.eye(3), "svs": np.array([2.0, 3.0, 4.0])}

        result = invert_weights(r, None, sigma2g=0.5, N=10.0, x=x, mode="Winv_ahat_h")

        assert np.isnan(result[1]).all()
        assert np.isfinite(result[0]).all()
        assert np.isfinite(result[2]).all()
