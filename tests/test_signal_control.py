"""Exercise signal-phase behavior without starting CARLA."""
import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from benchmark.navigation.signal_control import install_route_signal_detection, yellow_stopping_distance


def xyz(x=0, y=0, z=0):
    return NS(x=x, y=y, z=z)


def waypoint(x, lane_type='Driving', lane_id=-1):
    return NS(road_id=1, lane_id=lane_id, section_id=0, lane_type=lane_type, s=x,
              transform=NS(location=xyz(x)))


class SignalControl(unittest.TestCase):
    options = {'deceleration_mps2':2.5, 'reaction_time_s':.1, 'buffer_m':1.0}

    def setup_agent(self, yellow_stop, lane_type='Driving', signal_lane_id=-1):
        pose, velocity = xyz(90), xyz(4)
        light = NS(id=3, state='Yellow', get_opendrive_id=lambda:'100',
                   get_affected_lane_waypoints=lambda:[waypoint(100, lane_type, signal_lane_id)])
        light.get_state = lambda:light.state
        vehicle = NS(get_location=lambda:pose, get_velocity=lambda:velocity,
                     bounding_box=NS(extent=xyz(2)))
        agent = NS(_vehicle=vehicle, _lights_list=[light], _ignore_traffic_lights=False,
                   _base_tlight_threshold=5, _last_traffic_light=None,
                   _map=NS(get_waypoint_xodr=lambda road,lane,s:waypoint(s, lane_type, lane)))
        modules = {'carla':NS(LaneType=NS(Driving='Driving', Biking='Biking'),
                             TrafficLightState=NS(Red='Red',Yellow='Yellow',Green='Green')),
                   'agents.navigation.basic_agent':NS(TrafficLightDetectionResult=lambda a,b:(a,b))}
        with patch.dict(sys.modules, modules):
            install_route_signal_detection(agent, [(waypoint(0, lane_type),None),(waypoint(120, lane_type),None)], yellow_stop)
        return agent,light,pose,velocity

    def test_biking_reference_uses_existing_red_yellow_and_green_rules(self):
        agent, light, pose, velocity = self.setup_agent(self.options, lane_type='Biking')
        self.assertTrue(agent._affected_by_traffic_light(max_distance=9)[0])
        self.assertEqual(agent.route_signal_diagnostic['lane_id'], -1)
        self.assertEqual(agent.route_signal_diagnostic['signal_id'], '100')
        pose.x = 91
        velocity.x = 0
        light.state = 'Red'
        self.assertTrue(agent._affected_by_traffic_light(max_distance=5)[0])
        light.state = 'Green'
        self.assertFalse(agent._affected_by_traffic_light(max_distance=5)[0])

    def test_biking_route_does_not_borrow_another_lane_signal(self):
        agent, light, _, _ = self.setup_agent(self.options, lane_type='Biking', signal_lane_id=-2)
        light.state = 'Red'
        self.assertFalse(agent._affected_by_traffic_light(max_distance=9)[0])
        self.assertIsNone(agent.route_signal_diagnostic)

    def test_old_configuration_still_ignores_yellow(self):
        agent,light,_,_=self.setup_agent(None)
        self.assertFalse(agent._affected_by_traffic_light(max_distance=9)[0])
        light.state='Red'
        self.assertTrue(agent._affected_by_traffic_light(max_distance=9)[0])

    def test_yellow_stop_holds_when_speed_threshold_shrinks_and_releases_on_green(self):
        agent,light,pose,velocity=self.setup_agent(self.options)
        self.assertTrue(agent._affected_by_traffic_light(max_distance=9)[0])
        json.dumps(agent.route_signal_diagnostic, allow_nan=False)
        pose.x=91;velocity.x=0
        self.assertTrue(agent._affected_by_traffic_light(max_distance=5)[0])
        self.assertTrue(agent.route_signal_diagnostic['latched_yellow_stop'])
        light.state='Red'
        self.assertTrue(agent._affected_by_traffic_light(max_distance=5)[0])
        light.state='Green'
        self.assertFalse(agent._affected_by_traffic_light(max_distance=5)[0])

    def test_close_yellow_proceed_does_not_flip_when_vehicle_slows(self):
        agent,light,pose,velocity=self.setup_agent(self.options)
        pose.x=97
        self.assertFalse(agent._affected_by_traffic_light(max_distance=9)[0])
        self.assertFalse(agent.route_signal_diagnostic['yellow_decision']['stop'])
        velocity.x=0
        self.assertFalse(agent._affected_by_traffic_light(max_distance=5)[0])
        light.state='Green'
        agent._affected_by_traffic_light(max_distance=5)
        light.state='Yellow';pose.x=90;velocity.x=4
        self.assertTrue(agent._affected_by_traffic_light(max_distance=9)[0])

    def test_stopping_distance_has_explicit_units(self):
        self.assertAlmostEqual(yellow_stopping_distance(4, **self.options),4.6)
        self.assertAlmostEqual(yellow_stopping_distance(0, **self.options),1)


if __name__=='__main__':
    unittest.main()
