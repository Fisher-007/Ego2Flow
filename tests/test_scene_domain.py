"""Check route support and departure handling."""
import unittest

from benchmark.scenes.scene_domain import SceneDomain
from benchmark.runtime.run import incident_stop_reason


class SceneDomainTests(unittest.TestCase):
    def test_multiple_departures_are_recorded_without_removal_or_repeat_spam(self):
        domain=SceneDomain(reviewed_polygon=[[-10,-10],[10,-10],[10,10],[-10,10]])
        vehicles=[dict(request_id=i,position_m=[0,y,0]) for i,y in enumerate([11,12,0])]
        first=domain.new_vehicle_violations(vehicles)
        self.assertEqual([r['request_id'] for r in first],[0,1])
        self.assertEqual(len(vehicles),3)
        self.assertEqual(domain.new_vehicle_violations(vehicles),[])
        vehicles[2]['position_m']=[0,13,0]
        self.assertEqual(domain.new_vehicle_violations(vehicles)[0]['request_id'],2)

    def test_legal_corridor_failure_stops_when_requested_but_retains_observation_mode(self):
        # Apply the configured corridor-departure stop rule.
        self.assertEqual(incident_stop_reason([], 'left_legal_corridor', True), 'left_legal_corridor')
        self.assertIsNone(incident_stop_reason([], 'left_legal_corridor', False))
        self.assertEqual(incident_stop_reason([{'actor_id': 1}], 'exception', True), 'collision')
        self.assertIsNone(incident_stop_reason([], 'exception', True))

    def test_route_segment_crossing_defect_is_rejected_before_driving(self):
        issue=dict(id='pavement-seam',status='open',exclude_bbox_carla_xy_m=[-1,-1,1,1])
        routes={'route':dict(validated_waypoints=[dict(position_carla_m=[-5,0,0]),
                                                 dict(position_carla_m=[5,0,0])])}
        with self.assertRaisesRegex(ValueError,'pavement-seam'):
            SceneDomain([issue]).validate_routes(routes,['route'])

    def test_body_can_enter_defect_while_its_center_is_outside(self):
        domain=SceneDomain([dict(id='seam',status='open',exclude_bbox_carla_xy_m=[0,0,2,2])])
        vehicle=dict(request_id=7,position_m=[-1,1,1],
                     bounding_box_world_vertices_m=[[-3,0,0],[.5,0,0],[.5,2,0],[-3,2,0]])
        result=domain.first_vehicle_violation([vehicle])
        self.assertEqual((result['request_id'],result['map_issue_id']),(7,'seam'))
        self.assertEqual(vehicle['position_m'],[-1,1,1])

    def test_normal_lateral_motion_within_reviewed_area_is_preserved(self):
        domain=SceneDomain(reviewed_polygon=[[-10,-10],[10,-10],[10,10],[-10,10]])
        vehicle=dict(request_id=0,position_m=[0,4,0])
        self.assertIsNone(domain.first_vehicle_violation([vehicle]))
        vehicle['position_m']=[0,11,0]
        self.assertEqual(domain.first_vehicle_violation([vehicle])['reason'],'outside_reviewed_driving_area')

    def test_scene_without_boundaries_allows_vehicle_positions(self):
        self.assertIsNone(SceneDomain().first_vehicle_violation([dict(request_id=0,position_m=[0,0,0])]))

if __name__=='__main__':unittest.main()
