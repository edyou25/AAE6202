"""LQR controller for circular-path tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import B747Params, G


@dataclass(frozen=True)
class CircleRef:
    center_x: float
    center_y: float
    radius: float
    ccw: bool = True


@dataclass(frozen=True)
class ControlConfig:
    v_ref: float = 210.0
    dt: float = 0.05
    q_e_psi: float = 30.0
    q_e_phi: float = 10.0
    r_u: float = 1.0
    k_v: float = 120_000.0
    k_ct: float = 1.0 / 3500.0
    max_heading_correction_rad: float = np.deg2rad(25.0)


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve discrete LQR via Riccati iteration."""
    p = Q.copy()
    for _ in range(2000):
        bt_p = Bd.T @ p
        s = R + bt_p @ Bd
        k = np.linalg.solve(s, bt_p @ Ad)
        p_next = Ad.T @ p @ Ad - Ad.T @ p @ Bd @ k + Q
        if np.max(np.abs(p_next - p)) < 1e-10:
            p = p_next
            break
        p = p_next
    bt_p = Bd.T @ p
    return np.linalg.solve(R + bt_p @ Bd, bt_p @ Ad)


class LQRCircleController:
    def __init__(self, cfg: ControlConfig, p: B747Params):
        self.cfg = cfg
        self.p = p
        self.k_lqr = self._build_gain()

    def _build_gain(self) -> np.ndarray:
        v = self.cfg.v_ref
        tau = self.p.tau_phi

        # Inner-loop states x = [e_psi, e_phi]
        # e_psi_dot = (g/v)e_phi
        # e_phi_dot = -(1/tau)e_phi + (1/tau)u
        a = np.array(
            [
                [0.0, G / v],
                [0.0, -1.0 / tau],
            ],
            dtype=float,
        )
        b = np.array([[0.0], [1.0 / tau]], dtype=float)

        ad = np.eye(2) + self.cfg.dt * a
        bd = self.cfg.dt * b

        q = np.diag([self.cfg.q_e_psi, self.cfg.q_e_phi])
        r = np.array([[self.cfg.r_u]], dtype=float)
        return dlqr(ad, bd, q, r)

    def compute_control(self, state: np.ndarray, ref: CircleRef) -> tuple[dict, dict]:
        x, y, psi, phi, v = state

        dx = x - ref.center_x
        dy = y - ref.center_y
        radius_now = max(np.hypot(dx, dy), 1e-6)
        theta = np.arctan2(dy, dx)

        tangent = theta + (np.pi / 2.0 if ref.ccw else -np.pi / 2.0)
        e_ct = ref.radius - radius_now  # positive means inside

        # Outer-loop guidance: steer toward tangent corrected by radial error.
        heading_correction = np.arctan(self.cfg.k_ct * (-e_ct))
        heading_correction = float(
            np.clip(
                heading_correction,
                -self.cfg.max_heading_correction_rad,
                self.cfg.max_heading_correction_rad,
            )
        )
        psi_des = wrap_pi(tangent + heading_correction)

        turn_sign = 1.0 if ref.ccw else -1.0
        phi_ff = np.arctan((self.cfg.v_ref**2) / (G * ref.radius)) * turn_sign
        phi_ff = float(np.clip(phi_ff, -self.p.phi_limit_rad, self.p.phi_limit_rad))

        e_psi = wrap_pi(psi - psi_des)
        e_phi = wrap_pi(phi - phi_ff)

        err = np.array([e_psi, e_phi], dtype=float)
        delta_phi_cmd = float(-(self.k_lqr @ err.reshape(-1, 1)).item())
        phi_cmd = float(np.clip(phi_ff + delta_phi_cmd, -self.p.phi_limit_rad, self.p.phi_limit_rad))

        drag_guess = 0.5 * 1.225 * v * v * self.p.wing_area * self.p.cd0
        thrust = drag_guess + self.cfg.k_v * (self.cfg.v_ref - v)
        thrust = float(np.clip(thrust, 0.0, self.p.max_thrust))

        control = {"phi_cmd": phi_cmd, "thrust": thrust}
        info = {
            "e_ct": e_ct,
            "e_psi": e_psi,
            "e_phi": e_phi,
            "psi_des": psi_des,
            "phi_ff": phi_ff,
            "radius_now": radius_now,
        }
        return control, info
