import unittest

import numpy as np
import numpy.testing as npt

from visual import _build_frame_schedule, body_to_world, build_aircraft_point_cloud


class VisualTests(unittest.TestCase):
    def test_body_to_world_applies_rotation_and_translation(self) -> None:
        points = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

        transformed = body_to_world(points, x=10.0, y=-5.0, psi=np.pi / 2.0)

        expected = np.array([[10.0, -4.0], [9.0, -5.0]], dtype=float)
        npt.assert_allclose(transformed, expected, atol=1e-12)

    def test_build_aircraft_point_cloud_respects_scale(self) -> None:
        cloud_1x = build_aircraft_point_cloud(scale=1.0)
        cloud_2x = build_aircraft_point_cloud(scale=2.0)

        self.assertEqual(cloud_1x.wing.shape, cloud_2x.wing.shape)
        npt.assert_allclose(cloud_2x.fuselage, 2.0 * cloud_1x.fuselage)
        npt.assert_allclose(cloud_2x.h_tail, 2.0 * cloud_1x.h_tail)

    def test_build_frame_schedule_downsamples_to_requested_fps(self) -> None:
        time = np.linspace(0.0, 10.0, 1001)

        frame_ids, interval_ms, writer_fps = _build_frame_schedule(
            time=time,
            n_states=time.size,
            fps=20,
            max_frames=0,
        )

        self.assertEqual(frame_ids[0], 0)
        self.assertEqual(frame_ids[-1], time.size - 1)
        self.assertTrue(np.all(np.diff(frame_ids) > 0))
        self.assertLessEqual(writer_fps, 20)
        self.assertGreater(interval_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
