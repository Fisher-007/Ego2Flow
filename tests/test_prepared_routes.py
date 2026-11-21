import unittest

from benchmark.navigation.prepared_routes import remaining_route_index


class PreparedRouteTests(unittest.TestCase):
    def test_rolling_beyond_first_waypoint_radius(self):
        # Preparation can move the vehicle past the planner's waypoint pop radius.
        points=[(0.,0.),(2.,0.),(4.,0.),(6.,0.),(8.,0.),(10.,0.)]
        index=remaining_route_index(points,(5.8076,.35))
        self.assertEqual(index,3)
        self.assertEqual(points[index:],[(6.,0.),(8.,0.),(10.,0.)])

    def test_progress_follows_the_route_through_a_bend(self):
        points=[(0.,0.),(5.,0.),(5.,5.),(5.,10.)]
        self.assertEqual(remaining_route_index(points,(5.2,2.)),2)


if __name__=='__main__':
    unittest.main()
