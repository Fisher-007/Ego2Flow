"""Hand-calculated cases for traffic accounting and energy metrics."""
import json
import math
from pathlib import Path
import unittest

from benchmark.metrics.evaluation import PROFILE, GroupedSpeed, HardBraking, collision_episodes, request_delay
from benchmark.metrics.evaluation_geometry import EvaluationGeometry


def car(ident,x,y=0,z=.1,speed=5,acceleration=(0,0,0)):
    return dict(request_id=ident,actor_id=ident+10,route='lane0',position_m=[x,y,z],
        speed_mps=speed,velocity_mps=[speed,0,0],acceleration_mps2=list(acceleration),forward_xy=[1,0],
        bounding_box_world_vertices_m=[[x+dx,y+dy,z+dz] for dx in (-2,2) for dy in (-1,1) for dz in (-.5,.5)])


def route(y=0,z=0,free_flow=10):
    return dict(validated_waypoints=[dict(road_id=1,lane_id=-1,s_m=x,
        position_carla_m=[x,y,z],lane_center_forward_xy=[1,0],lane_width_m=4,
        free_flow_speed_mps=free_flow) for x in (0,50,100)],
        evaluation=dict(legal_route_names=['lane0','lane1']))


def geometry():
    routes=dict(lane0=route(),lane1=route(4),bridge=route(0,6))
    spec=dict(analysis_route_names=list(routes),analysis_polygon_carla_xy_m=[[-5,-5],[105,-5],[105,10],[-5,10]])
    return EvaluationGeometry(routes,spec)


class EightMetricTests(unittest.TestCase):
    def setUp(self):self.settings=json.loads(PROFILE.read_text())

    def test_group_means_do_not_create_false_speed_variability(self):
        groups=GroupedSpeed();groups.add('straight',10,10,20);groups.add('turn',3,3,10)
        self.assertEqual(groups.result()[0],0)
        groups.add('straight',0,0,20)
        self.assertAlmostEqual(groups.result()[0],math.sqrt(20))

    def test_delay_retains_waiting_unfinished_and_initial_remaining_time(self):
        requests=[dict(id=0,arrival_s=0,admitted_s=2,completed_s=12),
                  dict(id=1,arrival_s=5,admitted_s=None,completed_s=None),
                  dict(id=2,arrival_s=0,admitted_s=0,completed_s=None),
                  dict(id=3,arrival_s=25,admitted_s=None,completed_s=None)]
        mean,rows=request_delay(requests,[0,20],{0:8,2:5})
        self.assertAlmostEqual(mean,34/3);self.assertEqual(len(rows),3)
        self.assertEqual([r['delay_s'] for r in rows],[4,15,15])

    def test_legal_progress_rejects_offroad_reverse_and_super_freeflow_credit(self):
        g=geometry()
        for vehicle,expected in [(car(0,20,speed=50),1.),(car(0,20,speed=-5),0.),(car(0,20,y=8),0.)]:
            credit,_=g.legal_progress(vehicle,g.associations(vehicle))
            self.assertEqual(credit,expected)

    def test_lane_change_uses_both_occupied_lanes_but_only_one_ttc(self):
        g=geometry();vehicles=[car(0,20,y=2,speed=10),car(1,35,y=0,speed=5),car(2,30,y=4,speed=5),car(3,25,z=6.1,speed=0)]
        a={v['request_id']:g.associations(v) for v in vehicles}
        relation=g.following(vehicles,a,.1)[0]
        self.assertEqual(relation['leader_request_id'],2)
        self.assertAlmostEqual(relation['gap_m'],6)
        self.assertAlmostEqual(relation['ttc_s'],1.2)
        self.assertNotIn('bridge',a[0])

    def test_nonclosing_leader_remains_an_exposure_relation(self):
        g=geometry();vehicles=[car(0,20,speed=5),car(1,30,speed=8)]
        a={v['request_id']:g.associations(v) for v in vehicles}
        relation=g.following(vehicles,a,.1)[0]
        self.assertTrue(math.isinf(relation['ttc_s']))

    def test_rear_exit_clearance_remains_legal_but_failed_route_gives_no_credit(self):
        g=geometry();v=car(0,101,speed=10)
        self.assertEqual(g.legal_progress(v,g.associations(v))[0],1.)
        v['corridor']=dict(sustained_departure=True)
        self.assertEqual(g.legal_progress(v,g.associations(v))[0],0.)

    def test_distance_and_dwell_clip_to_same_domain(self):
        g=geometry();a,b=car(0,-10),car(0,0)
        pieces=g.interval_exposure(a,b,0,2,[0,2])
        self.assertEqual(pieces,[(.5,1.,5.)])

    def test_collision_double_callbacks_merge_but_separate_contact_remains(self):
        g=geometry()
        events=[dict(actor_id=a,other_actor_id=b,time_s=t,other_actor_type='vehicle.test',
                     sensor_position_m=[20,0,0],impulse=[10,0,0])
                for a,b,t in [(10,11,1),(11,10,1),(10,11,1.1),(10,11,2.5)]]
        result,missing=collision_episodes(events,g,[0,3],1.)
        self.assertEqual(len(result),2);self.assertEqual(result[0]['raw_callbacks'],3);self.assertEqual(missing,0)

    def test_braking_uses_signed_longitudinal_acceleration_and_duration(self):
        detector=HardBraking(self.settings)
        for i in range(10):detector.observe(car(0,20,acceleration=(0,5,0)),i*.05,(i+1)*.05)
        self.assertEqual(detector.result(),[])
        for i in range(6):detector.observe(car(1,20,acceleration=(-4,0,0)),i*.05,(i+1)*.05)
        self.assertEqual(len(detector.result()),1)
        for i in range(8):detector.observe(car(2,20,acceleration=(-20,0,0)),i*.05,(i+1)*.05,True)
        self.assertEqual(len(detector.result()),1)
        bounce=car(3,20,acceleration=(0,0,-100))
        bounce['velocity_mps'][2]=2.
        for i in range(10):detector.observe(bounce,i*.05,(i+1)*.05)
        self.assertEqual(len(detector.result()),1)




if __name__=='__main__':unittest.main()
