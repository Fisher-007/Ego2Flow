import unittest
from types import SimpleNamespace as NS

from benchmark.navigation.route_waypoints import native_endpoint_index, resolve_waypoint


class ExactNativeWaypoints(unittest.TestCase):
    def setUp(self):
        self.point = dict(road_id=233, lane_id=-1, s_m=12.95656092121851,
                          position_carla_m=[4., 5., 6.])
        self.endpoint = NS(road_id=233, section_id=0, lane_id=-1, s=self.point['s_m'],
            next_until_lane_end=lambda resolution: [],
            transform=NS(location=NS(x=4., y=5., z=6.)))
        self.calls = []
        def lookup(road, lane, station):
            self.calls.append((road, lane, station))
            return None
        self.native = NS(get_waypoint_xodr=lookup,
                         get_topology=lambda: [(self.endpoint, self.endpoint)])

    def test_rounding_failure_uses_identical_native_object_without_station_shift(self):
        actual = resolve_waypoint(self.native, self.point,
            exact_endpoints=native_endpoint_index(self.native))
        self.assertIs(actual, self.endpoint)
        self.assertEqual(self.calls, [(233, -1, self.point['s_m'])])

    def test_missing_exact_endpoint_does_not_select_nearby_lane_or_station(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved native route point'):
            resolve_waypoint(self.native, self.point, exact_endpoints={})
        self.assertEqual(len(self.calls), 1)

    def test_saved_endpoint_inside_larger_topology_edge_is_retained(self):
        entry = NS(road_id=233, section_id=0, lane_id=-1, s=0.,
            next_until_lane_end=lambda resolution: [self.endpoint])
        exit = NS(road_id=233, section_id=0, lane_id=-1, s=20.)
        self.native.get_topology = lambda: [(entry, exit)]
        actual = resolve_waypoint(self.native, self.point,
            exact_endpoints=native_endpoint_index(self.native))
        self.assertIs(actual, self.endpoint)
        self.assertEqual(self.calls, [(233, -1, self.point['s_m'])])

    def test_installed_geometry_check_remains_active(self):
        self.endpoint.transform.location.z += 1
        with self.assertRaisesRegex(ValueError, 'differs from installed map'):
            resolve_waypoint(self.native, self.point,
                exact_endpoints=native_endpoint_index(self.native))

    def test_existing_compatible_mode_still_resolves_its_inward_sample(self):
        self.native.get_waypoint_xodr = lambda road, lane, station: (
            self.endpoint if station < self.point['s_m'] else None)
        self.assertIs(resolve_waypoint(self.native, self.point), self.endpoint)

