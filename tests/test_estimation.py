import unittest

import numpy as np
import numpy.testing as npt

from dynamics import B747Params, rk4_step
from estimation import EstimationConfig, GaussianMAPEstimator


class EstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = B747Params()
        self.cfg = EstimationConfig()
        self.x0 = np.array([120_500.0, -500.0, np.deg2rad(95.0), np.deg2rad(2.0), 210.0], dtype=float)
        self.control = {"phi_cmd": np.deg2rad(3.0), "thrust": 250_000.0}
        self.dt = 0.1

    def test_covariance_helpers_return_diagonal_positive_matrices(self) -> None:
        for cov in (self.cfg.q_cov(), self.cfg.r_cov(), self.cfg.p0_cov()):
            npt.assert_allclose(cov, np.diag(np.diag(cov)))
            self.assertTrue(np.all(np.diag(cov) > 0.0))

    def test_step_with_perfect_measurement_matches_prediction(self) -> None:
        estimator = GaussianMAPEstimator(self.x0, self.cfg)
        x_prior = rk4_step(self.x0, self.control, self.dt, self.params)

        result = estimator.step(self.control, x_prior, self.dt, self.params)

        npt.assert_allclose(result["x_prior"], x_prior)
        npt.assert_allclose(result["x_post"], x_prior)
        self.assertAlmostEqual(result["map_objective"], 0.0, places=10)
        self.assertLess(np.trace(result["p_post"]), np.trace(result["p_prior"]))
        npt.assert_allclose(estimator.x_hat, x_prior)

    def test_step_moves_estimate_toward_measurement(self) -> None:
        estimator = GaussianMAPEstimator(self.x0, self.cfg)
        x_prior = rk4_step(self.x0, self.control, self.dt, self.params)
        measurement = x_prior + np.array([100.0, -50.0, 0.05, -0.03, 8.0], dtype=float)

        result = estimator.step(self.control, measurement, self.dt, self.params)

        self.assertLess(
            np.linalg.norm(measurement - result["x_post"]),
            np.linalg.norm(measurement - result["x_prior"]),
        )
        self.assertGreater(np.linalg.norm(result["innovation"]), 0.0)
        self.assertLess(np.trace(result["p_post"]), np.trace(result["p_prior"]))
        self.assertGreaterEqual(result["update_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
