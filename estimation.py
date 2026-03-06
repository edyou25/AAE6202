"""Bayes/MAP estimator for B747 simulation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from dynamics import B747Params, rk4_step


@dataclass(frozen=True)
class EstimationConfig:
    """Configuration for Gaussian Bayes/MAP estimation."""

    process_std: tuple[float, float, float, float, float] = (
        2.0,
        2.0,
        np.deg2rad(0.2),
        np.deg2rad(0.2),
        0.5,
    )
    measurement_std: tuple[float, float, float, float, float] = (
        40.0,
        40.0,
        np.deg2rad(1.0),
        np.deg2rad(1.0),
        2.0,
    )
    init_std: tuple[float, float, float, float, float] = (
        50.0,
        50.0,
        np.deg2rad(2.0),
        np.deg2rad(2.0),
        3.0,
    )

    def q_cov(self) -> np.ndarray:
        s = np.asarray(self.process_std, dtype=float)
        return np.diag(s * s)

    def r_cov(self) -> np.ndarray:
        s = np.asarray(self.measurement_std, dtype=float)
        return np.diag(s * s)

    def p0_cov(self) -> np.ndarray:
        s = np.asarray(self.init_std, dtype=float)
        return np.diag(s * s)


class GaussianMAPEstimator:
    """Simple Gaussian Bayes estimator (prior + likelihood -> posterior MAP)."""

    def __init__(self, x0: np.ndarray, cfg: EstimationConfig):
        self.cfg = cfg
        self.x_hat = np.asarray(x0, dtype=float).copy()
        self.p = cfg.p0_cov()
        self.q = cfg.q_cov()
        self.r = cfg.r_cov()
        self._i = np.eye(self.x_hat.size)

    def step(
        self,
        control: dict,
        measurement: np.ndarray,
        dt: float,
        p_dyn: B747Params,
    ) -> dict:
        """Run predict-update and return diagnostics for visualization."""
        t0 = perf_counter()

        # Prior (prediction): x^- from process model.
        x_prior = rk4_step(self.x_hat, control, dt, p_dyn)
        p_prior = self.p + self.q

        z = np.asarray(measurement, dtype=float)
        innovation = z - x_prior
        s_mat = p_prior + self.r

        k_gain = p_prior @ np.linalg.inv(s_mat)
        x_post = x_prior + k_gain @ innovation
        p_post = (self._i - k_gain) @ p_prior

        # MAP objective at posterior solution.
        d_prior = x_post - x_prior
        d_meas = z - x_post
        map_obj = 0.5 * float(d_prior.T @ np.linalg.solve(p_prior, d_prior))
        map_obj += 0.5 * float(d_meas.T @ np.linalg.solve(self.r, d_meas))

        self.x_hat = x_post
        self.p = p_post
        update_ms = (perf_counter() - t0) * 1000.0

        return {
            "x_prior": x_prior,
            "x_post": x_post,
            "p_prior": p_prior,
            "p_post": p_post,
            "innovation": innovation,
            "map_objective": map_obj,
            "update_ms": update_ms,
        }
