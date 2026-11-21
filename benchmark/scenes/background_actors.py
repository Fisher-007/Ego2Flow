"""Spawn the parked vehicles specified by the scene."""
import math
import numpy as np


def spawn_background(world, context, carla):
    actors, records = [], []
    try:
        for row in context['parked_actors']:
            spec = row['native_asset']
            matrix = np.asarray(spec['actor_transform_carla'])
            r = matrix[:3, :3]
            pose = carla.Transform(carla.Location(*matrix[:3, 3]), carla.Rotation(
                pitch=math.degrees(math.asin(float(np.clip(r[2, 0], -1, 1)))),
                yaw=math.degrees(math.atan2(r[1, 0], r[0, 0])),
                roll=math.degrees(math.atan2(-r[2, 1], r[2, 2]))))
            bp = world.get_blueprint_library().find(spec['blueprint'])
            bp.set_attribute('role_name', 'parked_context_'+row['track_uuid'])
            color = spec.get('color_rgb') or row.get('appearance', {}).get('color_rgb')
            if color and bp.has_attribute('color'):
                bp.set_attribute('color', ','.join(map(str, color)))
            staging = carla.Transform(carla.Location(pose.location.x, pose.location.y, pose.location.z+1000))
            actor = world.spawn_actor(bp, staging)
            actors.append(actor)
            actor.set_simulate_physics(False)
            actor.set_enable_gravity(False)
            actor.set_transform(pose)
            records.append({'source_track':row['track_uuid'],'actor_id':actor.id,'blueprint':actor.type_id,
                'role':'optional_parked_context','in_demand_ledger':False,
                'source_color_rgb':color, 'native_color':actor.attributes.get('color'),
                'asset_support_fit':spec.get('asset_support_fit'),
                'control_mode':'stationary_obstacle; removable runtime actor, not a baked map mesh'})
        return actors, records
    except Exception:
        for actor in reversed(actors):actor.destroy()
        raise
