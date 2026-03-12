import shutil
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from reporting import save_report_figures


class ReportingTests(unittest.TestCase):
    def test_save_report_figures_creates_report_assets(self) -> None:
        out_dir = Path("data/test_outputs/reporting")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = save_report_figures(output_dir=out_dir, t_end=2.0, dt=0.1, seed=7)

        self.assertEqual(
            set(paths),
            {"overview", "part1_control", "part2_dynamics", "part3_estimation"},
        )
        for saved in paths.values():
            saved_path = Path(saved)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
