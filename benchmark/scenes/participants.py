"""Apply vehicle defaults and record the spawned vehicle's properties."""


def apply_participant_defaults(config, requests):
    for request in requests:
        request.setdefault('vehicle_blueprint', config['vehicle_blueprint'])
        request.setdefault('target_speed_kmh', config['target_speed_kmh'])


def record_participant(request, vehicle):
    box = vehicle.bounding_box
    request['actor_type_id'] = vehicle.type_id
    request['actor_role_name'] = vehicle.attributes.get('role_name')
    request['actor_bounding_box'] = {
        'location_m': [box.location.x, box.location.y, box.location.z],
        'extent_m': [box.extent.x, box.extent.y, box.extent.z],
        'rotation_deg': [box.rotation.roll, box.rotation.pitch, box.rotation.yaw],
    }

