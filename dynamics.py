"""Boeing 747 planar flight dynamics with RK4 integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


G = 9.81
RHO = 1.225


@dataclass(frozen=True)
class B747Params:
    """Representative Boeing 747-400 parameters for a planar simulation."""

    mass: float = 396_890.0          # kg
    wing_area: float = 510.97        # m^2
    cd0: float = 0.022               # parasitic drag coefficient
    k_induced: float = 0.048         # induced drag factor
    max_thrust: float = 1_126_290.0  # N (total)
    tau_phi: float = 2.0             # s, roll response time constant
    phi_limit_rad: float = np.deg2rad(35.0)
    v_min: float = 120.0             # m/s
    v_max: float = 320.0             # m/s


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def b747_dynamics(state: np.ndarray, control: Dict[str, float], p: B747Params) -> np.ndarray:
    """Continuous-time dynamics for planar turning flight.

    state = [x, y, psi, phi, v]
      x,y : inertial position (m)
      psi : heading (rad)
      phi : bank angle (rad)
      v   : airspeed (m/s)

    control keys:
      phi_cmd: commanded bank angle (rad)
      thrust : total thrust (N)
    """
    x, y, psi, phi, v = state
    _ = (x, y)

    phi_cmd = _clip(control["phi_cmd"], -p.phi_limit_rad, p.phi_limit_rad)
    thrust = _clip(control["thrust"], 0.0, p.max_thrust)

    v_safe = _clip(v, p.v_min, p.v_max)

    # Quasi-steady level-flight lift coefficient with bank penalty.
    cos_phi = max(np.cos(phi), 0.2)
    cl = (2.0 * p.mass * G) / (RHO * v_safe * v_safe * p.wing_area * cos_phi)
    cd = p.cd0 + p.k_induced * cl * cl
    drag = 0.5 * RHO * v_safe * v_safe * p.wing_area * cd

    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    psi_dot = G * np.tan(phi) / max(v_safe, 1.0)
    phi_dot = (phi_cmd - phi) / p.tau_phi
    v_dot = (thrust - drag) / p.mass

    return np.array([x_dot, y_dot, psi_dot, phi_dot, v_dot], dtype=float)


def rk4_step(state: np.ndarray, control: Dict[str, float], dt: float, p: B747Params) -> np.ndarray:
    """One RK4 integration step."""
    k1 = b747_dynamics(state, control, p)
    k2 = b747_dynamics(state + 0.5 * dt * k1, control, p)
    k3 = b747_dynamics(state + 0.5 * dt * k2, control, p)
    k4 = b747_dynamics(state + dt * k3, control, p)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
