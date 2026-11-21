"""Prime the cold native renderer before any traffic requests are admitted."""
import queue


def prime_cameras(world, carla_map, carla, specifications, position):
    """Warm up the CARLA renderer with temporary cameras."""
    if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Renderer priming requires an empty traffic world')
    pose=carla_map.get_waypoint(carla.Location(*position)).transform
    original=world.get_settings()
    cameras=[]
    try:
        settings=world.get_settings();settings.synchronous_mode=False
        world.apply_settings(settings)
        for spec in specifications:
            if spec['type']!='sensor.camera.rgb':continue
            bp=world.get_blueprint_library().find(spec['type'])
            attributes=dict(sensor_tick=0,**spec.get('blueprint_attributes',{}))
            attributes.update(image_size_x=spec['width'],image_size_y=spec['height'],fov=spec['fov'])
            for key,value in attributes.items():bp.set_attribute(key,str(value))
            relative=carla.Location(x=spec['x'],y=spec['y'],z=spec['z'])
            location=pose.transform(relative)
            transform=carla.Transform(carla.Location(x=location.x,y=location.y,z=location.z),carla.Rotation(
                pitch=pose.rotation.pitch+spec['pitch'],yaw=pose.rotation.yaw+spec['yaw'],
                roll=pose.rotation.roll+spec['roll']))
            sensor=world.spawn_actor(bp,transform);stream=queue.Queue()
            sensor.listen(stream.put);cameras.append((spec['id'],sensor,stream))
        first=world.get_snapshot().frame
        # Upstream LEAD AgentWrapper explicitly skips the first ten ticks:
        # native sensors may omit images during this startup interval.
        for _ in range(10):world.wait_for_tick(30.)
        received={}
        for name,_,stream in cameras:
            image=stream.get(timeout=60)
            while True:
                try:image=stream.get_nowait()
                except queue.Empty:break
            received[name]=dict(frame=image.frame,width=image.width,height=image.height)
        return dict(discarded_empty_world_ticks=10,first_frame=first,last_frame=world.get_snapshot().frame,
            camera_frames=received,images_used_by_model=False,traffic_vehicles_present=False,
            mode='asynchronous empty-world camera initialization; synchronous settings restored before vehicle creation',
            scope='Cold-renderer startup only. Actual attached model sensors still have their own physical history and exact-frame checks.')
    finally:
        for _,sensor,_ in cameras:
            sensor.stop();sensor.destroy()
        world.apply_settings(original)
        if original.synchronous_mode:world.tick(1800.)

