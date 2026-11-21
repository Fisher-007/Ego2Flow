"""Entry clearance follows the vehicle footprint, including parallel lanes."""
import math
from shapely.geometry import Polygon, MultiPoint


def buffered_spawn_polygon(transform,bounding_box,longitudinal_gap_m):
    """Build the spawn clearance polygon from the vehicle's bounding box."""
    yaw=math.radians(transform.rotation.yaw);fx,fy=math.cos(yaw),math.sin(yaw)
    cx=transform.location.x+fx*bounding_box.location.x-fy*bounding_box.location.y
    cy=transform.location.y+fy*bounding_box.location.x+fx*bounding_box.location.y
    half_length=bounding_box.extent.x+longitudinal_gap_m
    half_width=bounding_box.extent.y+.1
    return Polygon([(cx+fx*x-fy*y,cy+fy*x+fx*y) for x,y in
        [(-half_length,-half_width),(half_length,-half_width),(half_length,half_width),(-half_length,half_width)]])


def entry_is_clear(transform,active,initial_population=False):
    if not active:return True
    body=next(iter(active.values()))['vehicle'].bounding_box
    spawn=buffered_spawn_polygon(transform,body,.1 if initial_population else 2.)
    for item in active.values():
        vehicle=item['vehicle']
        pose=item['birth_transform'] if item.get('needs_initial_record') else vehicle.get_transform()
        if abs(pose.location.z-transform.location.z)>3.:continue
        occupied=MultiPoint([(p.x,p.y) for p in vehicle.bounding_box.get_world_vertices(pose)]).convex_hull
        if spawn.intersects(occupied):return False
    return True
