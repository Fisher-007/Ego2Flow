"""Apply the external leader speed pulse."""
import math


def target_speed_mps(spec, time_s):
    """Raised-cosine deceleration, hold, recovery, then the original setpoint."""
    base = spec['base_speed_mps']
    depth = spec['speed_drop_fraction']
    t = time_s - spec['start_s']
    down, hold, up = (spec[k] for k in ('deceleration_s', 'hold_s', 'recovery_s'))
    if t <= 0 or t >= down + hold + up:
        fraction = 0.
    elif t < down:
        fraction = .5 * (1 - math.cos(math.pi * t / down))
    elif t < down + hold:
        fraction = 1.
    else:
        fraction = .5 * (1 + math.cos(math.pi * (t-down-hold) / up))
    return base * (1 - depth * fraction)


def external_ids(config):
    return {r['id'] for r in config.get('request_manifest', []) if r.get('external_intervention')}


def evaluated_ids(config, requests):
    external = external_ids(config)
    return [r['id'] for r in requests if r['id'] not in external and (
        config['deployment'] == 'H' or
        (config['deployment'] == 'M' and r['mixed_role'] == 'evaluated'))]


def validate(spec, requests):
    ids = {r['id'] for r in requests if r.get('external_intervention')}
    if ids != {spec['request_id']}:
        raise ValueError('External pulse must address the one declared intervention leader')
    if not (0 < spec['speed_drop_fraction'] < 1 and spec['base_speed_mps'] > 0
            and all(spec[k] > 0 for k in ('deceleration_s', 'hold_s', 'recovery_s'))):
        raise ValueError('External leader pulse needs a positive speed and smooth positive durations')
