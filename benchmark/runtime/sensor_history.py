"""Collect sensor history during physical vehicle preparation."""
import math
import queue
from benchmark.runtime.control_submission import submit_commands, control_echo


def prepare_physical_history(client, world, active, dt, carla, settling_s=1.5, rolling_s=1.1, native_policy=False, after_tick=None, defer_policy_until_motion=False, history_start_s=0., require_ready_before_motion=False):
    """Collect sensor history before the observation window opens."""
    settling_steps=math.ceil(settling_s/dt)
    rolling_steps=math.ceil(rolling_s/dt)
    history_start_steps=math.ceil(history_start_s/dt)
    if not 0 <= history_start_steps < settling_steps:
        raise ValueError('History capture must start within physical settling')
    history_readers={rid:getattr(item.get('policy'),'input_history_status',None)
                     for rid,item in active.items()}
    if require_ready_before_motion:
        available=(settling_steps-history_start_steps-1)*dt
        for rid,reader in history_readers.items():
            if reader is not None and available+1e-6 < reader()['required_s']:
                raise ValueError(f'Request {rid}: only {available:.3f}s real history before motion')
    ready_at_release={}
    traces=[]
    checked = mismatched = 0
    for step in range(settling_steps+rolling_steps):
        rolling=step>=settling_steps
        if step==settling_steps:
            for rid,reader in history_readers.items():
                if reader is not None:
                    ready_at_release[rid]=reader()
                    if require_ready_before_motion and not ready_at_release[rid]['ready']:
                        raise RuntimeError(f'Request {rid}: actual history not ready at motion release')
        commands=[]
        applied=carla.VehicleControl() if rolling else carla.VehicleControl(brake=1.)
        for item in active.values():
            vehicle=item['vehicle']
            native = item['request'].get('native_policy_preparation', native_policy)
            actual_applied = item.get('preparation_control', carla.VehicleControl()) if rolling and native else applied
            item['preparation_applied'] = actual_applied
            if rolling and item.get('is_tm'):
                item['agent'].run_step()
            else:
                commands.append(carla.command.ApplyVehicleControl(vehicle.id, actual_applied))
            if rolling and (not native or step==settling_steps):
                commands.append(carla.command.ApplyTargetVelocity(vehicle.id,
                    vehicle.get_transform().get_forward_vector()*item['request']['initial_speed_mps']))
        submit_commands(client,commands)
        frame=world.tick()
        if after_tick is not None:
            after_tick(world.get_snapshot(), (step+1-settling_steps-rolling_steps)*dt)
        row=dict(time_before_observation_s=(step+1-settling_steps-rolling_steps)*dt,
                 frame=frame,stage='physical_rolling' if rolling else 'settling',vehicles=[])
        for item in active.values():
            vehicle=item['vehicle']
            if vehicle.is_dormant:
                raise RuntimeError(f"Prepared actor {item['request']['id']} is dormant: shared-world active range is insufficient")
            pose=vehicle.get_transform();velocity=vehicle.get_velocity()
            if rolling and item.get('is_tm'):
                from benchmark.runtime.control_submission import control_state
                actual_control, matches = control_state(vehicle.get_control()), None
            else:
                actual_control, matches = control_echo(vehicle, item['preparation_applied'])
                checked += 1
                mismatched += int(not matches)
            speed=velocity.dot(pose.get_forward_vector())
            diagnostic=None
            if item['is_visual']:
                images={}
                for key,stream in item['policy_queues'].items():
                    try:
                        value=stream.get(timeout=60)
                        while value.frame<frame:
                            value=stream.get(timeout=60)
                    except queue.Empty as error:
                        raise RuntimeError(f"Preparation sensor timeout: request {item['request']['id']}, "
                            f"sensor {key}, frame {frame}, step {step}, rolling={rolling}") from error
                    if value.frame!=frame:
                        raise RuntimeError('Initial-history sensor frame differs from world frame')
                    images[key]=value
                item['policy_images']=images;item['policy_input_frame']=frame
                observer=getattr(item['policy'],'observe_inputs',None)
                if observer is not None and step>=history_start_steps:
                    observer(images,preparation=True)
                    item.setdefault('history_capture_first_frame',frame)
                # Use the final preparation frame for the first observation decision.
                if (step+1==settling_steps and not defer_policy_until_motion) or (rolling and step+1<settling_steps+rolling_steps):
                    policy=item['policy']
                    gps=pose.transform(carla.Location(x=policy.localization_sensor_x_m))
                    command,diagnostic=policy.step(images,speed,
                        [pose.location.x,pose.location.y,pose.location.z],pose.rotation.yaw,
                        gps_position_xyz=[gps.x,gps.y,gps.z])
                    if rolling and item['request'].get('native_policy_preparation',native_policy):
                        item['preparation_control']=carla.VehicleControl(**command)
                    elif rolling:
                        policy.processor.control=carla.VehicleControl()
            if rolling and not item['is_visual'] and not item.get('is_tm') and item['request'].get('native_policy_preparation',native_policy):
                if item['request'].get('external_intervention'):
                    item['agent'].set_target_speed(3.6*item['request']['initial_speed_mps'])
                item['preparation_control']=item['agent'].run_step()
            row['vehicles'].append(dict(request_id=item['request']['id'],speed_mps=speed,
                applied_control=actual_control,control_roundtrip_match=matches,
                position_m=[pose.location.x,pose.location.y,pose.location.z],
                input_frame=frame if item['is_visual'] else None,
                model_step=diagnostic.get('model_step') if diagnostic else None,
                policy_status=diagnostic.get('status') if diagnostic else None,
                model_called=diagnostic.get('model_called') if diagnostic else None,
                history_duration_s=diagnostic.get('history_duration_s') if diagnostic else None,
                history_capture_enabled=step>=history_start_steps,
                input_history=history_readers[item['request']['id']]() if history_readers[item['request']['id']] is not None else None,
                actual_gear=vehicle.get_control().gear,
                history_frames=diagnostic.get('lidar_history_frames') if diagnostic else None))
        traces.append(row)
    states=[]
    for item in active.values():
        request=item['request'];vehicle=item['vehicle'];pose=vehicle.get_transform()
        speed=vehicle.get_velocity().dot(pose.get_forward_vector())
        anchor=request['time_zero_anchor_position_m'];location=[pose.location.x,pose.location.y,pose.location.z]
        states.append(dict(request_id=request['id'],actor_id=vehicle.id,
            target_speed_mps=request['initial_speed_mps'],measured_speed_mps=speed,
            speed_error_mps=speed-request['initial_speed_mps'],position_m=location,
            requested_time_zero_anchor_m=anchor,horizontal_anchor_error_m=math.dist(location[:2],anchor[:2])))
        item['initial_prepared']=True;request['initial_velocity_applied_s']=0.
    from benchmark.navigation.prepared_routes import align_prepared_background_routes
    aligned=align_prepared_background_routes(active)
    return dict(method='native_policy_history' if native_policy else 'physical_rolling_with_native_history',
        prepared_background_routes=aligned,
        control_submission=dict(method='checked_batch_sync',checked_vehicle_frames=checked,
            mismatched_vehicle_frames=mismatched),
        scope=('One initial velocity application then assigned policy with real sensor history; anchor is a preparation target, not a forced time-zero state.' if native_policy else
            'Prescribed physical preparation; actual sensor history, no teleportation. Per-request native_policy_preparation overrides repeated velocity forcing when present.'),
        settling_s=settling_steps*dt,rolling_initialization_s=rolling_steps*dt,
        history_start_s=history_start_steps*dt,
        require_ready_before_motion=require_ready_before_motion,
        input_history_at_motion_release=ready_at_release,
        defer_policy_until_motion=defer_policy_until_motion,
        last_preparation_frame=frame,states_at_time_zero=states,trace=traces)


def observe_admitted_history(item, images, elapsed, exclude_settling=False):
    """Collect admitted-vehicle history after the optional spawn-settling interval."""
    observer=getattr(item['policy'],'observe_inputs',None)
    if observer is None:
        return False
    if (exclude_settling and not item.get('initial_prepared')
            and elapsed-item['request']['admitted_s'] < 1.-1e-6):
        return False
    observer(images,preparation=False)
    if images:
        item.setdefault('history_capture_first_frame', next(iter(images.values())).frame)
    return True
