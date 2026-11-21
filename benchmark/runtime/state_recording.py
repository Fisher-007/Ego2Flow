"""Record road geometry and initial vehicle state for scoring."""
import math


def road_fields(waypoint, velocity, forward):
    tangent=waypoint.transform.get_forward_vector()
    horizontal=math.hypot(tangent.x,tangent.y)
    speed=math.hypot(velocity.x,velocity.y)
    dx,dy=(velocity.x/speed,velocity.y/speed) if speed>.1 else (forward.x,forward.y)
    grade=tangent.z/horizontal*(dx*tangent.x+dy*tangent.y)/horizontal
    return dict(road_grade=grade,road_surface_height_m=waypoint.transform.location.z,
        road_tangent_xyz=[tangent.x,tangent.y,tangent.z],
        native_lane=dict(road_id=waypoint.road_id,lane_id=waypoint.lane_id,
            section_id=waypoint.section_id,is_junction=waypoint.is_junction))


def initial_state(item, carla_map, time_s, prepared=False):
    vehicle=item['vehicle']
    transform=vehicle.get_transform() if prepared else item['birth_transform']
    location=transform.location;forward=transform.get_forward_vector()
    velocity=vehicle.get_velocity();acceleration=vehicle.get_acceleration()
    waypoint=carla_map.get_waypoint(location)
    corners=vehicle.bounding_box.get_world_vertices(transform)
    state=dict(request_id=item['request']['id'],actor_id=vehicle.id,route=item['request']['route'],
        position_m=[location.x,location.y,location.z],velocity_mps=[velocity.x,velocity.y,velocity.z],
        acceleration_mps2=[acceleration.x,acceleration.y,acceleration.z],
        speed_mps=math.hypot(velocity.x,velocity.y),forward_xy=[forward.x,forward.y],
        bounding_box_world_vertices_m=[[p.x,p.y,p.z] for p in corners],
        road_surface_offset_m=location.z-waypoint.transform.location.z,
        **road_fields(waypoint,velocity,forward))
    return dict(time_s=time_s,vehicle=state,source='post_preparation_state' if prepared else 'admission_before_first_control')
