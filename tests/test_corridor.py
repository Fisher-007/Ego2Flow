import unittest
from benchmark.navigation.corridor import Corridor, CorridorTracker
from benchmark.metrics.route_outcomes import evaluate


def points(y):
    return [{'position_carla_m':[x,y,0], 'lane_center_forward_xy':[1,0], 'lane_width_m':3.5} for x in [0,5,10]]


def vehicle(x, y, forward=1):
    return {'request_id':0, 'position_m':[x,y,0], 'forward_xy':[forward,0],
            'bounding_box_world_vertices_m':[[x+dx,y+dy,0] for dx in [-1,1] for dy in [-.8,.8]]}


class CorridorTests(unittest.TestCase):
    def tracker(self):
        return CorridorTracker(Corridor({'center_margin_m':.2,'sustained_departure_s':1.,
            'paths':[{'id':'supplied','validated_waypoints':points(0)},
                     {'id':'branch','validated_waypoints':points(3.5)}]}))

    def test_neighbor_exit_succeeds_despite_representative_lane_departure(self):
        t=self.tracker()
        for x in [1,2,3,4,8,9]:t.update(vehicle(x,3.5),.5)
        v=vehicle(10.1,3.5);v['corridor']=t.update(v,.05)
        self.assertEqual(v['corridor']['forward_exit_crossing'],'branch')
        self.assertFalse(v['corridor']['sustained_departure'])
        result=evaluate([{'id':0,'arrival_s':0,'route':'ego','completed_s':6,'first_off_route_s':1,'exit_path':'branch'}],
            [{'time_s':6,'vehicles':[v]}],{'ego':points(0)},10,task='road_corridor_and_exit')
        self.assertEqual(result['passed_bounded_route_check'],1)

    def test_reverse_exit_does_not_complete(self):
        t=self.tracker();t.update(vehicle(9,0,-1),.5)
        result=t.update(vehicle(10.1,0,-1),.5)
        self.assertIsNone(result['forward_exit_crossing'])
        self.assertTrue(result['sustained_departure'])

    def test_leave_corridor_cannot_erase_violation_by_reentering(self):
        t=self.tracker();t.update(vehicle(2,0),.5)
        for x in [3,4]:t.update(vehicle(x,15),.5)
        self.assertTrue(t.update(vehicle(5,0),.5)['sustained_departure'])

    def test_crossing_other_branch_after_commitment_is_invalid(self):
        t=self.tracker()
        for x in [1,2,3]:t.update(vehicle(x,0),.5)
        for x in [7,8,9]:t.update(vehicle(x,3.5),.5)
        r=t.update(vehicle(10.1,3.5),.05)
        self.assertTrue(r['sustained_departure'])
        self.assertFalse(r['exit_on_compatible_path'])

    def test_wide_tight_bend_has_valid_cross_section_envelope(self):
        from benchmark.navigation.route_lane import lane_corridor
        bend=[{'position_carla_m':[36.5183,16.3430,0], 'lane_center_forward_xy':[.176431,.984090], 'lane_width_m':6.66927},
              {'position_carla_m':[36.3633,17.3133,0], 'lane_center_forward_xy':[-.278235,.960375], 'lane_width_m':6.66927}]
        _, polygon, _, _, _=lane_corridor(bend)
        from shapely.geometry import Point
        self.assertTrue(polygon.is_valid)
        self.assertTrue(polygon.covers(Point(36.44,16.83)))

    def test_multilane_permitted_transition_and_rear_terminal(self):
        a,b=points(0),points(3.5)
        for ps,lane in [(a,-1),(b,-2)]:
            for p in ps:p.update(road_id=1,section_id=0,lane_id=lane,permitted_lane_change_to=[[1,0,-2]])
        data=dict(task='multilane_movement_and_terminal',entry_path='a',terminal_reference='rear',
            center_margin_m=.2,sustained_departure_s=1.,paths=[dict(id='a',validated_waypoints=a),dict(id='b',validated_waypoints=b)])
        tracker=CorridorTracker(Corridor(data))
        tracker.update(vehicle(2,0),.5);tracker.update(vehicle(4,1.7),.5)
        self.assertFalse(tracker.update(vehicle(6,3.5),.5)['sustained_departure'])
        self.assertIsNone(tracker.update(vehicle(10.5,3.5),.5)['forward_exit_crossing'])
        self.assertTrue(tracker.update(vehicle(11.1,3.5),.5)['exit_on_compatible_path'])
        for p in a:p['permitted_lane_change_to']=[]
        tracker=CorridorTracker(Corridor(data));tracker.update(vehicle(2,0),.5)
        self.assertTrue(tracker.update(vehicle(5,3.5),.5)['sustained_departure'])

    def test_overlapping_junction_routes_do_not_invent_lane_change(self):
        a,b=points(0),points(.00002)
        for ps,road in [(a,1),(b,2)]:
            for p in ps:p.update(road_id=road,section_id=0,lane_id=-1)
        tracker=CorridorTracker(Corridor(dict(task='multilane_movement_and_terminal',
            entry_path='a',terminal_reference='rear',center_margin_m=.2,sustained_departure_s=1.,
            paths=[dict(id='a',validated_waypoints=a),dict(id='b',validated_waypoints=b)])))
        for x,y in [(1,0),(3,.00002),(6,0),(9,.00002)]:
            self.assertFalse(tracker.update(vehicle(x,y),.05)['sustained_departure'])
        self.assertTrue(tracker.update(vehicle(11.1,0),.05)['exit_on_compatible_path'])
