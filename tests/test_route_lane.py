import unittest
from benchmark.navigation.route_lane import evaluate


def frame(y=0, x=5, t=1):
    return {'time_s':t,'dt_s':.05,'frame':int(t/.05),'vehicles':[{'request_id':0,
        'position_m':[x,y,0],'route_offset_m':abs(y),'speed_mps':1,
        'bounding_box_world_vertices_m':[[x+dx,y+dy,0] for dx in [-2,2] for dy in [-1,1]]}]}


class LaneFootprint(unittest.TestCase):
    def setUp(self):
        self.points=[{'position_carla_m':[x,0,0],'lane_center_forward_xy':[1,0],
            'lane_width_m':4,'is_junction':False} for x in range(21)]

    def test_partial_footprint_outside_before_center_crosses(self):
        r=evaluate([frame(1.5)],self.points)
        self.assertAlmostEqual(r['max_outside_area_m2'],2)
        self.assertEqual(r['center_outside_seconds'],0)
        self.assertEqual(r['footprint_outside_seconds'],.05)

    def test_inside_and_junction_endpoints_are_distinct(self):
        self.assertEqual(evaluate([frame()],self.points)['max_outside_area_m2'],0)
        self.points[5]['is_junction']=True
        r=evaluate([frame(3),frame(3,x=1,t=2)],self.points)
        self.assertEqual(r['evaluated_frames'],0)
        self.assertEqual(r['excluded_endpoint_or_junction_frames'],2)
