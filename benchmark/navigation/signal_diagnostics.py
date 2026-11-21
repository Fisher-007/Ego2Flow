"""Read CARLA signal geometry and record BasicAgent decisions."""
import types
from benchmark.navigation.signal_control import upstream_signal_waypoint


def xyz(point):
    return [point.x, point.y, point.z]


def waypoint_record(wp):
    if wp is None:
        return None
    return {"road_id": wp.road_id, "lane_id": wp.lane_id, "s_m": wp.s,
            "is_junction": wp.is_junction, "lane_type": str(wp.lane_type),
            "position_m": xyz(wp.transform.location), "forward": xyz(wp.transform.get_forward_vector())}


def signal_record(actor):
    return {"actor_id": actor.id, "signal_id": actor.get_opendrive_id(), "state": str(actor.get_state())} if actor else None


def inspect_native_signals(world, carla_map=None):
    """Record native signal geometry and predicted overlap boxes."""
    from agents.tools.misc import get_trafficlight_trigger_location
    import carla
    if carla_map is None:
        carla_map = world.get_map()
    records = []
    for actor in world.get_actors().filter('traffic.traffic_light*'):
        transform, box = actor.get_transform(), actor.trigger_volume
        references, generated = actor.get_affected_lane_waypoints(), []
        for wp in references:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            center = upstream_signal_waypoint(carla_map, wp)
            generated.append({"center_waypoint": waypoint_record(center),
                "half_extents_forward_right_up_m": [1.5, max(.01, wp.lane_width / 4), 1.0],
                "source": "TrafficLightComponent.cpp InitializeSign formula, not queried UE component geometry"})
        records.append({**signal_record(actor), "actor_position_m": xyz(transform.location),
            "actor_yaw_deg": transform.rotation.yaw,
            "legacy_trigger_local_center_m": xyz(box.location), "legacy_trigger_extent_m": xyz(box.extent),
            "legacy_trigger_world_center_m": xyz(transform.transform(box.location)),
            "basic_agent_projected_trigger_waypoint": waypoint_record(carla_map.get_waypoint(get_trafficlight_trigger_location(actor))),
            "get_stop_waypoints": [waypoint_record(wp) for wp in actor.get_stop_waypoints()],
            "affected_lane_waypoints": [waypoint_record(wp) for wp in references],
            "generated_component_box_predictions": generated})
    return {"map": carla_map.name, "signals": records,
        "scope": "Read-only native API plus source-derived box positions; actual overlap evidence comes from vehicle.get_traffic_light in frames."}


def observe_agent(agent):
    """Record BasicAgent's traffic light and obstacle detections."""
    agent.signal_diagnostic = {}
    original_light = agent._affected_by_traffic_light
    original_vehicle = agent._vehicle_obstacle_detected

    def light(self, *args, **kwargs):
        result = original_light(*args, **kwargs)
        self.signal_diagnostic['traffic_light'] = {'detected': bool(result[0]),
                                                  'light': signal_record(result[1])}
        return result

    def vehicle(self, *args, **kwargs):
        result = original_vehicle(*args, **kwargs)
        self.signal_diagnostic['vehicle'] = {'detected': bool(result[0]),
            'actor_id': result[1].id if result[1] is not None else None, 'distance_m': result[2]}
        return result
    agent._affected_by_traffic_light = types.MethodType(light, agent)
    agent._vehicle_obstacle_detected = types.MethodType(vehicle, agent)


def vehicle_signal_state(vehicle, agent=None):
    return {"native_is_at_traffic_light": vehicle.is_at_traffic_light(),
            "native_traffic_light_state": str(vehicle.get_traffic_light_state()),
            "native_traffic_light": signal_record(vehicle.get_traffic_light()),
            "basic_agent_hazards": dict(agent.signal_diagnostic) if agent is not None else None,
            "route_signal_detection": getattr(agent, "route_signal_diagnostic", None),
            "hazard_clock": "control-time before tick; native overlap state is frame-end"}
