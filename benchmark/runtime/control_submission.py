"""Submit and acknowledge vehicle controls before the synchronous world tick."""


def submit_commands(client, commands):
    if not commands:
        return
    responses = client.apply_batch_sync(commands, False)
    if len(responses) != len(commands):
        raise RuntimeError("CARLA did not acknowledge every submitted command")
    for index, response in enumerate(responses):
        if response.has_error():
            raise RuntimeError(f"CARLA command {index} failed: {response.error}")


def control_state(control):
    result = {key: getattr(control, key) for key in (
        'throttle', 'steer', 'brake', 'hand_brake', 'reverse', 'manual_gear_shift')}
    if control.manual_gear_shift:
        result['gear'] = control.gear
    return result


def control_echo(vehicle, requested):
    """Read after world.tick(), while this client still owns the sync clock."""
    actual = control_state(vehicle.get_control())
    return actual, actual == control_state(requested)
