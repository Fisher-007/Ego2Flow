import math
from benchmark.policies.rule_control import idm_acceleration, TrafficManagerAgent, IDMSpeedReference, tm_path_locations

def test_idm_equilibrium_and_closing():
    # At v=6,v0=10 the equilibrium gap follows the exact free-speed term.
    gap=(2+6*1.5)/math.sqrt(1-(6/10)**4)
    assert abs(idm_acceleration(6,10,gap,6))<1e-12
    assert idm_acceleration(6,10,gap,3)<0
    assert idm_acceleration(0,10)==1.2
    assert idm_acceleration(0,10,2,0)==0
    assert idm_acceleration(6,10,0,0)==-math.inf


def test_idm_speed_servo_keeps_acceleration_and_releases_braking_backlog():
    reference=IDMSpeedReference()
    for _ in range(20):
        target=reference.step(6.,1.2,10.,.05)
    assert abs(target-7.2)<1e-10
    # An actual 5 m/s car must brake immediately, even if its old target was 7.2.
    assert reference.step(5.,-2.,10.,.05)==4.9
    assert reference.step(5.,-math.inf,10.,.05)==0.

def test_tm_owns_control_and_supplied_path():
    from types import SimpleNamespace
    calls=[]
    class Manager:
        def __getattr__(self,name):
            return lambda *a:calls.append((name,a))
    v=SimpleNamespace(set_autopilot=lambda *a:calls.append(('autopilot',a)))
    class Location:
        def __init__(self,x):self.x=x
        def distance(self,other):return abs(self.x-other.x)
    a=TrafficManagerAgent(v,Manager(),32190,30.,{'path_spacing_m':20.})
    locations=[Location(i) for i in range(0,46,2)]
    points=[(SimpleNamespace(road_id=1,transform=SimpleNamespace(location=p)),None) for p in locations]
    a.set_global_plan(points)
    assert not calls
    assert a.run_step() is None
    assert ('autopilot',(True,32190)) in calls
    assert any(n=='set_path' and [p.x for p in args[1]]==[0,20,40,44] for n,args in calls)
    assert not any(n=='apply_control' for n,_ in calls)
    a.disable()
    assert calls[-1]==('autopilot',(False,32190))
    assert not a.enabled


def test_mixed_identity_does_not_depend_on_method_or_include_leader():
    from benchmark.scenes.leader_pulse import evaluated_ids
    requests=[dict(id=i,mixed_role='evaluated' if i in [2,5,9] else 'background',external_intervention=i==0)
              for i in range(11)]
    for controller in ['Traffic Manager','IDM-based','LEAD']:
        cfg=dict(controller=controller,request_manifest=requests,deployment='M')
        assert evaluated_ids(cfg,requests)==[2,5,9]
        cfg['deployment']='H'
        assert evaluated_ids(cfg,requests)==list(range(1,11))


def test_tm_repairs_wrong_planned_road_once_without_overriding_control():
    from types import SimpleNamespace
    class Location:
        def __init__(self,x):self.x=x;self.y=0.;self.z=0.
        def distance(self,other):return abs(self.x-other.x)
    class Waypoint:
        road_id=1;lane_id=-1;section_id=0;is_junction=False
        def __init__(self,x):self.transform=SimpleNamespace(location=Location(x))
        def next(self,_):
            return [SimpleNamespace(road_id=i,section_id=0,lane_id=-1)
                    for i in ([2,3] if self.transform.location.x==28 else [1])]
    calls=[]
    class Manager:
        def __getattr__(self,name):return lambda *a:calls.append((name,a))
        def get_all_actions(self,v):return [('LaneFollow',SimpleNamespace(road_id=2984,is_junction=False))]
    v=SimpleNamespace(get_location=lambda:Location(18),set_autopilot=lambda *a:calls.append(('autopilot',a)))
    a=TrafficManagerAgent(v,Manager(),32190,30.,dict(path_spacing_m=20.,junction_lanes='route'))
    plan=[(Waypoint(x),None) for x in range(0,61,2)];a.set_global_plan(plan)
    a.run_step();a.run_step()
    imports=[args[1] for n,args in calls if n=='set_path']
    assert len(imports)==2
    assert [p.x for p in imports[-1]]==[18,38,58,60]
    assert len(a.navigation_updates)==1 and a.navigation_updates[0]['unexpected_planned_roads']==[2984]
    assert not any(n=='apply_control' for n,_ in calls)


def test_tm_inspects_neighbor_fork_after_legal_lane_change():
    from types import SimpleNamespace
    class Location:
        def __init__(self,x):self.x=x;self.y=0.;self.z=0.
        def distance(self,other):return abs(self.x-other.x)
    class Waypoint:
        road_id=1;section_id=0;is_junction=False;lane_type='Driving'
        def __init__(self,x,lane=-1):
            self.lane_id=lane;self.transform=SimpleNamespace(location=Location(x))
        def get_left_lane(self):return None
        def get_right_lane(self):return Waypoint(self.transform.location.x,-2)
        def next(self,_):
            roads=[2,3] if self.lane_id==-2 and self.transform.location.x==28 else [1]
            return [SimpleNamespace(road_id=i,section_id=0,lane_id=self.lane_id) for i in roads]
    calls=[]
    class Manager:
        def __getattr__(self,name):return lambda *a:calls.append((name,a))
        def get_all_actions(self,v):return [('LaneFollow',SimpleNamespace(road_id=2984,is_junction=False))]
    v=SimpleNamespace(get_location=lambda:Location(18),set_autopilot=lambda *a:None)
    plan=[(Waypoint(x),None) for x in range(0,61,2)]
    route_only=TrafficManagerAgent(v,Manager(),32190,30.,dict(path_spacing_m=20.,junction_lanes='route'))
    route_only.set_global_plan(plan);route_only.run_step();assert not route_only.navigation_updates
    adjacent=TrafficManagerAgent(v,Manager(),32190,30.,dict(path_spacing_m=20.,junction_lanes='route_and_adjacent'))
    adjacent.set_global_plan(plan);adjacent.run_step()
    assert adjacent.navigation_updates[0]['unexpected_planned_roads']==[2984]
    assert not any(n=='apply_control' for n,_ in calls)
