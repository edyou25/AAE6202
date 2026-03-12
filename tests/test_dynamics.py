import unittest

import numpy as np

from dynamics import B747Params, G, RHO, b747_dynamics, rk4_step


def _trim_drag(v: float, p: B747Params, phi: float = 0.0) -> float:
    cos_phi = max(np.cos(phi), 0.2)
    cl = (2.0 * p.mass * G) / (RHO * v * v * p.wing_area * cos_phi)
    cd = p.cd0 + p.k_induced * cl * cl
    return 0.5 * RHO * v * v * p.wing_area * cd


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = B747Params()

    def test_b747_dynamics_applies_input_clipping(self) -> None:
        state = np.array([0.0, 0.0, 0.0, 0.0, 210.0], dtype=float)
        control = {"phi_cmd": 10.0, "thrust": 2.0 * self.params.max_thrust}

        dstate = b747_dynamics(state, control, self.params)
        expected_drag = _trim_drag(210.0, self.params)

        self.assertAlmostEqual(dstate[0], 210.0)
        self.assertAlmostEqual(dstate[1], 0.0)
        self.assertAlmostEqual(dstate[2], 0.0)
        self.assertAlmostEqual(dstate[3], self.params.phi_limit_rad / self.params.tau_phi)
        self.assertAlmostEqual(
            dstate[4],
            (self.params.max_thrust - expected_drag) / self.params.mass,
        )

    def test_rk4_step_preserves_trimmed_straight_flight(self) -> None:
        dt = 0.2
        speed = 200.0
        state = np.array([1_000.0, -2_000.0, 0.0, 0.0, speed], dtype=float)
        control = {"phi_cmd": 0.0, "thrust": _trim_drag(speed, self.params)}

        next_state = rk4_step(state, control, dt, self.params)

        self.assertAlmostEqual(next_state[0], state[0] + speed * dt, places=6)
        self.assertAlmostEqual(next_state[1], state[1], places=6)
        self.assertAlmostEqual(next_state[2], state[2], places=6)
        self.assertAlmostEqual(next_state[3], state[3], places=6)
        self.assertAlmostEqual(next_state[4], state[4], places=6)

    def test_rk4_step_turns_when_bank_is_held(self) -> None:
        dt = 0.1
        phi = np.deg2rad(10.0)
        state = np.array([0.0, 0.0, 0.0, phi, 210.0], dtype=float)
        control = {"phi_cmd": phi, "thrust": _trim_drag(210.0, self.params, phi=phi)}

        next_state = rk4_step(state, control, dt, self.params)

        self.assertGreater(next_state[2], state[2])
        self.assertAlmostEqual(next_state[3], state[3], places=6)


if __name__ == "__main__":
    unittest.main()
