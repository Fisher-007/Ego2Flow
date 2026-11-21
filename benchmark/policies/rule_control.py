"""CARLA Traffic Manager and IDM-based driving methods."""
import math

IDM_DEFAULTS = dict(time_headway_s=1.5, minimum_gap_m=2., acceleration_mps2=1.2,
                    comfortable_deceleration_mps2=2., exponent=4.)


def idm_acceleration(speed, desired_speed, gap=None, leader_speed=0., parameters=None):
    p = {**IDM_DEFAULTS, **(parameters or {})}
    free = 1. - (speed / desired_speed) ** p['exponent']
    if gap is None:
        return p['acceleration_mps2'] * free
    if gap <= 0:
        return -math.inf  # Apply full braking on contact or overlap.
    dynamic = speed*p['time_headway_s'] + speed*(speed-leader_speed)/(2*math.sqrt(
        p['acceleration_mps2']*p['comfortable_deceleration_mps2']))
    desired_gap = p['minimum_gap_m'] + max(0., dynamic)
    return p['acceleration_mps2']*(free-(desired_gap/gap)**2)


class IDMSpeedReference:
    """Integrate IDM acceleration into a target speed for CARLA's PID controller."""
    def __init__(self):
        self.speed = None

    def step(self, actual_speed, acceleration, desired_speed, dt):
        base = actual_speed if self.speed is None else self.speed
        if acceleration < 0.:
            base = min(base, actual_speed)
        self.speed = max(0., min(desired_speed, base + acceleration*dt))
        return self.speed


def tm_path_locations(plan, spacing_m):
    """Sample route locations for Traffic Manager to follow."""
    locations=[w.transform.location for w,_ in plan]
    distances=[0.]
    for previous,current in zip(locations,locations[1:]):
        distances.append(distances[-1]+previous.distance(current))
    selected=[0];last=0.
    for i,distance in enumerate(distances[1:],1):
        if distance-last>=spacing_m:
            selected.append(i);last=distance
    selected=sorted(set([*selected,len(locations)-1]))
    return [locations[i] for i in selected]


class TrafficManagerAgent:
    autonomous = True
    def __init__(self, vehicle, manager, port, speed_kmh, options):
        self.vehicle, self.manager, self.port = vehicle, manager, port
        self.signal_diagnostic = {}
        self.enabled = False
        self.plan = []
        self.options = options
        self.junction_lanes = options.get('junction_lanes')
        if self.junction_lanes is None and options.get('path_adapter_version'):
            path_format = options['path_adapter_version']
            family, _, number = path_format.rpartition('_v')
            if family == 'native_tm_route_tail_refresh':
                self.junction_lanes = {'4': 'route', '5': 'route_and_adjacent'}.get(number)
            elif path_format.removesuffix('-v1') == 'native-cache-forward-outlets':
                self.junction_lanes = 'route_and_adjacent'
        self.speed_kmh = speed_kmh
        self.navigation_updates = []

    def set_global_plan(self, plan, **kwargs):
        self.plan = list(plan)
        self.stations=[0.]
        for (a,_),(b,_) in zip(self.plan,self.plan[1:]):
            self.stations.append(self.stations[-1]+a.transform.location.distance(b.transform.location))
        self.branches=[];self.refreshed_branches=set()
        if self.junction_lanes in ('route', 'route_and_adjacent'):
            for i,(waypoint,_) in enumerate(self.plan[:-1]):
                lanes=[waypoint]
                if self.junction_lanes == 'route_and_adjacent':
                    for side in ('left','right'):
                        neighbor=getattr(waypoint,'get_'+side+'_lane')()
                        if (neighbor is not None and neighbor.road_id==waypoint.road_id
                                and neighbor.lane_id*waypoint.lane_id>0
                                and neighbor.lane_type==waypoint.lane_type):
                            lanes.append(neighbor)
                if any(len({(w.road_id,w.section_id,w.lane_id) for w in lane.next(2.1)})>1 for lane in lanes):
                    if not self.branches or self.stations[i]-self.stations[self.branches[-1]]>3.:
                        self.branches.append(i)
        self.route_roads={w.road_id for w,_ in self.plan}
        if self.enabled:
            self.manager.set_path(self.vehicle, tm_path_locations(self.plan,self.options['path_spacing_m']), True)

    def enable(self):
        if self.enabled:
            return
        self.vehicle.set_autopilot(True, self.port)
        self.manager.set_desired_speed(self.vehicle, self.speed_kmh)
        self.manager.distance_to_leading_vehicle(self.vehicle, self.options.get('following_distance_m', 2.))
        self.manager.auto_lane_change(self.vehicle, True)
        self.manager.ignore_lights_percentage(self.vehicle, 0.)
        self.manager.ignore_signs_percentage(self.vehicle, 0.)
        self.manager.ignore_vehicles_percentage(self.vehicle, 0.)
        self.enabled = True
        self.set_global_plan(self.plan)

    def set_target_speed(self, speed):
        self.speed_kmh = speed
        if self.enabled:
            self.manager.set_desired_speed(self.vehicle, speed)

    def disable(self):
        """Unregister before destroying an actor or releasing synchronous TM."""
        if self.enabled:
            self.vehicle.set_autopilot(False, self.port)
            self.enabled = False

    def refresh_branch_navigation(self):
        if not self.branches:return
        position=self.vehicle.get_location()
        current=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(position))
        near=[i for i in self.branches if i not in self.refreshed_branches
              and 0.<=self.stations[i]-self.stations[current]<=12.]
        if not near:return
        actions=self.manager.get_all_actions(self.vehicle)
        unexpected=[w.road_id for _,w in actions if not w.is_junction and w.road_id not in self.route_roads]
        if not unexpected:return
        # Overlapping junction links can redirect ImportPath. Refresh the
        # remaining task route before reaching the fork.
        tail=self.plan[current:]
        self.manager.set_path(self.vehicle,tm_path_locations(tail,self.options['path_spacing_m']),True)
        self.refreshed_branches.add(near[0])
        self.navigation_updates.append(dict(route_index=current,branch_index=near[0],
            position_m=[position.x,position.y,position.z],unexpected_planned_roads=unexpected,
            remaining_road_ids=list(dict.fromkeys(w.road_id for w,_ in tail))))

    def run_step(self):
        self.enable()
        self.refresh_branch_navigation()
        return None  # TM alone submits control at the synchronous tick


def install_idm(agent, dt, options):
    """Use IDM for longitudinal control and BasicAgent for steering and traffic light detection."""
    from shapely.geometry import LineString, Point
    import types
    parameters = {k: options.get(k, v) for k, v in IDM_DEFAULTS.items()}
    original_set_speed = agent.set_target_speed
    agent.idm_desired_speed_mps = agent._target_speed / 3.6
    agent.idm_diagnostic = None
    agent.idm_speed_reference = IDMSpeedReference()

    def set_speed(self, kmh):
        self.idm_desired_speed_mps = kmh/3.6
        original_set_speed(kmh)

    def step(self):
        vehicle = self._vehicle
        velocity = vehicle.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        plan = list(self.get_local_planner().get_plan())
        points = [(w.transform.location.x,w.transform.location.y) for w,_ in plan]
        position = vehicle.get_location()
        # Include the actual current front of the route; the planner may have
        # consumed nearby points already. Geometry behind the ego is not a leader.
        line = LineString([(position.x,position.y), *points])
        own_front = max(line.project(Point(p.x,p.y)) for p in vehicle.bounding_box.get_world_vertices(vehicle.get_transform()))
        candidates = sorted((v for v in self._world.get_actors().filter('vehicle.*') if v.id != vehicle.id),
                            key=lambda v: v.get_location().distance(position))
        gap = None; front_speed = 0.; front_id = None
        for other in candidates:
            if other.get_location().distance(position)>80.:break
            hit, _, _ = self._vehicle_obstacle_detected([other],80.)
            if not hit:continue
            center = line.project(Point(other.get_location().x,other.get_location().y))
            if center <= own_front:continue
            rear = min(line.project(Point(p.x,p.y)) for p in other.bounding_box.get_world_vertices(other.get_transform()))
            distance = rear-own_front
            if gap is None or distance<gap:
                gap=distance;front_id=other.id
                v=other.get_velocity();forward=vehicle.get_transform().get_forward_vector()
                front_speed=max(0.,v.x*forward.x+v.y*forward.y)
        red, light = self._affected_by_traffic_light(self._lights_list,80.)
        signal = self.route_signal_diagnostic
        if red and signal:
            stop_gap=signal['reference_route_station_m']-signal['vehicle_route_station_m']-vehicle.bounding_box.extent.x
            if gap is None or stop_gap<gap:
                gap=stop_gap;front_speed=0.;front_id='light:'+str(light.id)
        acceleration=idm_acceleration(speed,self.idm_desired_speed_mps,gap,front_speed,parameters)
        next_speed=self.idm_speed_reference.step(speed,acceleration,self.idm_desired_speed_mps,dt)
        self.get_local_planner().set_speed(next_speed*3.6)
        control=self.get_local_planner().run_step()
        if next_speed<.02:
            control.throttle=0.;control.brake=max(control.brake,.5)
        self.idm_diagnostic=dict(desired_speed_mps=self.idm_desired_speed_mps,front_id=front_id,
            net_gap_m=gap,front_speed_mps=front_speed,acceleration_mps2=acceleration if math.isfinite(acceleration) else None,
            overlap_full_brake=not math.isfinite(acceleration),pid_target_speed_mps=next_speed)
        return control
    agent.set_target_speed=types.MethodType(set_speed,agent)
    agent.run_step=types.MethodType(step,agent)
    return parameters
