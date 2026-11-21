"""Check the vehicle's height against the road surface when an offset is available."""

ROAD_HEIGHT_TOLERANCE_M = 2.0


def road_sample_valid(vehicle):
    offset = vehicle.get('road_surface_offset_m')
    return offset is None or abs(offset) <= ROAD_HEIGHT_TOLERANCE_M

