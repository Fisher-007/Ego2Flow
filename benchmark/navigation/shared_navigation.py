"""Sample navigation routes and select each vehicle's starting point."""
from bisect import bisect_right


def sample_route_indices(route, spacing_m):
    """Select navigation points from a dense CARLA route."""
    selected = []
    distance_m = 0.0
    previous = None
    lane_changes = (5, 6)  # CARLA RoadOption.CHANGELANELEFT and CHANGELANERIGHT
    for i, (transform, option) in enumerate(route):
        command = option.value
        command_changed = command != previous and previous not in lane_changes
        if i == 0 or i == len(route) - 1 or command in lane_changes or command_changed or distance_m > spacing_m:
            selected.append(i)
            distance_m = 0.0
        else:
            distance_m += transform.location.distance(route[i - 1][0].location)
        previous = command
    return selected


def suffix_indices(sampled_indices, dense_stations_m, start_station_m):
    sampled_stations=[dense_stations_m[index] for index in sampled_indices]
    before=max(0,bisect_right(sampled_stations,start_station_m)-1)
    before=min(before,len(sampled_indices)-2)
    return sampled_indices[before:]
