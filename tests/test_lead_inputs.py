import math
import unittest

import numpy as np
from benchmark.policies.lead import decode_radar, local_position_to_gps


class LEADInputs(unittest.TestCase):
    def test_local_projection_signs_and_independent_inverse(self):
        radius = 6378137.
        for xyz in [(143.9, 193.2, 2.3), (-149., -10.7, 0.), (10000., -5000., 7.)]:
            lat, lon, z = local_position_to_gps(xyz)
            recovered = [radius*math.radians(lon),
                         -radius*math.log(math.tan(math.pi/4+math.radians(lat)/2)), z]
            np.testing.assert_allclose(recovered, xyz, atol=1e-7, rtol=0)
        self.assertGreater(local_position_to_gps((0., -10., 0.))[0], 0.)

    def test_native_radar_layout_preserves_velocity_and_angles(self):
        raw = np.array([[3., .2, -.1, 20.], [-2., -.4, .3, 7.]], dtype=np.float32)
        converted = decode_radar(raw.tobytes())
        np.testing.assert_array_equal(converted, raw[:, [3, 2, 1, 0]])
        depth, altitude, azimuth, velocity = converted[0]
        self.assertEqual(velocity, 3.)
        self.assertAlmostEqual(float(depth*np.cos(azimuth)*np.cos(altitude)), 19.5034, places=3)
