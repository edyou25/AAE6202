import unittest

import numpy as np

from controller import CircleRef, ControlConfig, LQRCircleController, dlqr, wrap_pi
from dynamics import B747Params, G


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = B747Params()
        self.cfg = ControlConfig(v_ref=210.0, dt=0.05)
        self.ref = CircleRef(center_x=0.0, center_y=0.0, radius=120_000.0, ccw=True)
        self.controller = LQRCircleController(self.cfg, self.params)

    def test_wrap_pi_returns_principal_values(self) -> None:
        self.assertAlmostEqual(wrap_pi(0.0), 0.0)
        self.assertAlmostEqual(wrap_pi(np.pi), -np.pi)
        self.assertAlmostEqual(wrap_pi(1.5 * np.pi), -0.5 * np.pi)
        self.assertAlmostEqual(wrap_pi(-1.5 * np.pi), 0.5 * np.pi)

    def test_dlqr_stabilizes_closed_loop(self) -> None:
        ad = np.array([[1.0, 0.1], [0.0, 1.0]])
        bd = np.array([[0.0], [0.1]])
        q = np.diag([10.0, 1.0])
        r = np.array([[1.0]])

        gain = dlqr(ad, bd, q, r)
        eigvals = np.linalg.eigvals(ad - bd @ gain)

        self.assertEqual(gain.shape, (1, 2))
        self.assertTrue(np.all(np.abs(eigvals) < 1.0))

    def test_compute_control_matches_trim_on_reference_circle(self) -> None:
        phi_ff = np.arctan((self.cfg.v_ref**2) / (G * self.ref.radius))
        state = np.array([self.ref.radius, 0.0, np.pi / 2.0, phi_ff, self.cfg.v_ref], dtype=float)

        control, info = self.controller.compute_control(state, self.ref)

        self.assertAlmostEqual(info["e_ct"], 0.0, places=6)
        self.assertAlmostEqual(info["e_psi"], 0.0, places=6)
        self.assertAlmostEqual(info["e_phi"], 0.0, places=6)
        self.assertAlmostEqual(control["phi_cmd"], info["phi_ff"], places=6)
        self.assertGreaterEqual(control["thrust"], 0.0)
        self.assertLessEqual(control["thrust"], self.params.max_thrust)

    def test_compute_control_respects_command_limits(self) -> None:
        state = np.array([self.ref.radius - 50_000.0, 0.0, 0.0, -self.params.phi_limit_rad, 100.0], dtype=float)

        control, info = self.controller.compute_control(state, self.ref)
        tangent = np.pi / 2.0
        heading_correction = wrap_pi(info["psi_des"] - tangent)

        self.assertLessEqual(abs(heading_correction), self.cfg.max_heading_correction_rad + 1e-12)
        self.assertLessEqual(abs(control["phi_cmd"]), self.params.phi_limit_rad + 1e-12)
        self.assertGreaterEqual(control["thrust"], 0.0)
        self.assertLessEqual(control["thrust"], self.params.max_thrust)


if __name__ == "__main__":
    unittest.main()
