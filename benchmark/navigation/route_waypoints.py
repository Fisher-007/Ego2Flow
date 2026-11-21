"""Resolve native road/lane/station references with endpoint rounding."""
import math


def native_endpoint_index(native, sampling_resolution=1.0):
    """Index native lane samples for saved route endpoints."""
    index = {}
    def lane_key(wp):
        return wp.road_id, wp.section_id, wp.lane_id
    for entry, exit in native.get_topology():
        samples = entry.next_until_lane_end(sampling_resolution)
        path = [wp for wp in samples if lane_key(wp) == lane_key(entry)]
        if lane_key(entry) == lane_key(exit):
            low, high = sorted((entry.s, exit.s))
            path = [wp for wp in path if low < wp.s < high]
        for wp in [entry] + path + [exit]:
            index[(wp.road_id, wp.lane_id, wp.s)] = wp
    return index


def resolve_waypoint(native, point, *, exact_endpoints=None):
    road, lane, station = int(point['road_id']), int(point['lane_id']), float(point['s_m'])
    waypoint = native.get_waypoint_xodr(road, lane, station)
    if waypoint is None and exact_endpoints is not None:
        waypoint = exact_endpoints.get((road, lane, station))
    if waypoint is None and exact_endpoints is None:
        # Probe 1 mm on either side to allow for endpoint rounding.
        for delta in (-.001, .001):
            waypoint = native.get_waypoint_xodr(road, lane, station + delta)
            if waypoint is not None:
                break
    if waypoint is None:
        raise ValueError(f'Unresolved native route point road={road} lane={lane} s={station}')
    p = waypoint.transform.location
    distance = math.dist((p.x, p.y, p.z), point['position_carla_m'])
    if distance > .01:
        raise ValueError(f'Native route differs from installed map: road={road} lane={lane} s={station}, distance={distance:.6f}m')
    return waypoint
