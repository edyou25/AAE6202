import shutil
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from controller import ControlConfig
from run import _backend_supports_show, _switch_to_backend, plot_results, simulate


class RunModuleTests(unittest.TestCase):
    def test_backend_supports_show_matches_backend_type(self) -> None:
        self.assertFalse(_backend_supports_show("Agg"))
        self.assertFalse(_backend_supports_show("module://matplotlib_inline.backend_inline"))
        self.assertTrue(_backend_supports_show("TkAgg"))

    def test_switch_to_backend_uses_requested_backend(self) -> None:
        switched = _switch_to_backend("Agg")
        self.assertEqual(switched.lower(), "agg")

    def test_simulate_supports_short_horizon(self) -> None:
        cfg = ControlConfig(v_ref=210.0, dt=0.1)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = simulate(
            t_end=1.0,
            cfg=cfg,
            seed=123,
        )

        expected_n = int(1.0 / cfg.dt) + 1
        self.assertEqual(time.shape, (expected_n,))
        self.assertEqual(hist.shape, (expected_n, 5))
        self.assertEqual(e_ct.shape, (expected_n,))
        self.assertEqual(e_psi_deg.shape, (expected_n,))
        self.assertEqual(phi_cmd_deg.shape, (expected_n,))
        self.assertEqual(telemetry["dstate"].shape, (expected_n, 5))
        self.assertEqual(telemetry["est_state_hat"].shape, (expected_n, 5))

    def test_plot_results_saves_png(self) -> None:
        cfg = ControlConfig(v_ref=210.0, dt=0.1)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref, _ = simulate(
            t_end=1.0,
            cfg=cfg,
            seed=123,
        )

        out_dir = Path("data/test_outputs/run")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary.png"

        saved = plot_results(time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref, out_path=out_path)

        self.assertEqual(Path(saved), out_path)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
