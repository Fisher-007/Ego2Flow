"""Start a background planner at the end of physical preparation."""
from bisect import bisect_right
import math

from shapely.geometry import LineString, Point


def remaining_route_index(points_xy, position_xy):
    """Find the next route waypoint after vehicle preparation."""
    stations=[0.]
    for a,b in zip(points_xy,points_xy[1:]):
        stations.append(stations[-1]+math.dist(a,b))
    progress=LineString(points_xy).project(Point(position_xy))
    return min(bisect_right(stations,progress),len(points_xy)-1)


def align_prepared_background_routes(active):
    records=[]
    for item in active.values():
        # Traffic Manager advances its own route buffer during preparation.
        if item['is_visual'] or item.get('is_tm'):
            continue
        plan=item['route']['navigation_plan']
        points=[(wp.transform.location.x,wp.transform.location.y) for wp,_ in plan]
        position=item['vehicle'].get_location()
        index=remaining_route_index(points,(position.x,position.y))
        item['agent'].set_global_plan(list(plan[index:]),
            stop_waypoint_creation=True,clean_queue=True)
        records.append(dict(request_id=item['request']['id'],skipped_waypoints=index,
            remaining_waypoints=len(plan)-index,
            scope='Only the prefix already traversed during physical preparation is removed; route, controller and observed pose are unchanged.'))
    return records
