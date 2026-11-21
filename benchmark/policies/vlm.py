"""OneVL and Alpamayo CARLA wrappers with separate inference processes."""
import json
import os
from pathlib import Path
import socket
import subprocess
from multiprocessing.connection import Connection
import numpy as np
from benchmark import ROOT
from benchmark.policies.trajectory_tracker import TrajectoryTracker, WAYPOINT_INTERVAL_S
from benchmark.policies.vlm_inputs import CalibratedRig,CausalHistory


def create(config):return VLMPolicy(config)


def navigation_text(route,index,position):
    end=min(len(route),index+100)
    index=min(range(index,end),key=lambda i:np.linalg.norm(route[i][0]-position))
    distance=0.;previous=position
    commands={1:'Turn left',2:'Turn right',3:'Continue straight through the junction',5:'Change to the left lane',6:'Change to the right lane'}
    for point,command in route[index:]:
        distance+=np.linalg.norm(point-previous);previous=point
        if distance>60:break
        if command in commands:return index,commands[command]+f' in {distance:.0f} metres; continue along the assigned road.'
    return index,'Continue along the current road following the assigned route.'


class VLMPolicy:
    def __init__(self,config):
        self.config=config;self.kind=config['model_kind'];self.next_session=0
        self.rig=CalibratedRig(json.loads(Path(config['calibration']).read_text())['calibration']);self.sensors=self.rig.sensors
        self.log_dir=Path(config['diagnostics_dir']);self.log_dir.mkdir(parents=True,exist_ok=True)
        left,right=socket.socketpair();self.connection=Connection(left.detach())
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (self.log_dir/'worker.log').open('a') as log:
            self.worker=subprocess.Popen([config['inference_python'],'-m','benchmark.policies.vlm_worker','--fd',str(right.fileno())],
                cwd=ROOT,env=env,pass_fds=(right.fileno(),),stdout=log,stderr=subprocess.STDOUT)
        right.close();self.connection.send(config);ready=self.receive(600)
        self.description=dict(name=config['name'],model=ready['description'],sensors=self.sensors,
            camera_calibration=config['calibration'],
            navigation='Assigned-route instruction and distance to the next maneuver',
            input='Camera frames and ego pose history',
            inference_period_s=.5,control_period_s=.05,
            tracker='Ego2Flow trajectory PID',
            startup='Brake until sensor history is ready.',
            invalid_output='Brake until the next scheduled prediction.')
        (self.log_dir/'description.json').write_text(json.dumps(self.description,indent=2))

    def receive(self,timeout=600):
        if not self.connection.poll(timeout):raise TimeoutError('VLM process timed out; inspect '+str(self.log_dir/'worker.log'))
        result=self.connection.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def infer(self,request):self.connection.send(request);return self.receive()

    def new_vehicle(self):
        session=VLMSession(self,self.next_session);self.next_session+=1;return session

    def close(self):
        if getattr(self,'worker',None) is not None:
            if self.worker.poll() is None:
                self.connection.send(None)
                try:self.worker.wait(timeout=20)
                except subprocess.TimeoutExpired:self.worker.terminate();self.worker.wait(timeout=20)
            self.connection.close();self.worker=None


class VLMSession:
    localization_sensor_x_m=0.
    def __init__(self,owner,session_id):self.owner=owner;self.session_id=session_id;self.sensors=owner.sensors

    def bind_request(self,request_id):self.request_id=int(request_id)

    def reset(self,plan):
        self.history=CausalHistory()
        self.tracker = TrajectoryTracker()
        self.route=[(np.array([w.transform.location.x,w.transform.location.y]),int(option.value))for w,option in plan]
        self.route_index=0;self.last_inference_s=None;self.inference_index=0;self.plan=None;self.last_output={}

    def input_history_status(self):
        poses=self.history.poses
        return dict(ready=self.history.ready(self.owner.kind),
            required_s=self.history.required_seconds(self.owner.kind),
            duration_s=poses[-1][0]-poses[0][0] if poses else 0.,
            first_frame=poses[0][1] if poses else None,
            last_frame=poses[-1][1] if poses else None)

    def observe_inputs(self,inputs,preparation=False):
        image=inputs['ring_front_center']
        self.history.append(image.frame,float(image.timestamp),self.owner.rig.pose(inputs),inputs)
        self.preparation=preparation

    def step(self,images,speed_mps,position_xyz,yaw_deg,gps_position_xyz=None):
        # The runner observes every matched frame, including physical settling.
        self.observe_inputs(images,getattr(self,'preparation',False))
        now,frame,pose=self.history.poses[-1]
        diagnostic=dict(model_step=self.inference_index,input_frame=frame,history_duration_s=now-self.history.poses[0][0],input_speed_mps=float(speed_mps),preparation=self.preparation)
        if not self.history.ready(self.owner.kind):
            return dict(steer=0.,throttle=0.,brake=1.),dict(diagnostic,status='waiting_for_actual_history',model_called=False)
        if self.last_inference_s is None or now-self.last_inference_s>=.5-1e-5:
            request=self.history.inputs(self.owner.kind,self.owner.rig)
            self.route_index,nav=navigation_text(self.route,self.route_index,np.asarray(position_xyz[:2]))
            request.update(navigation=nav,request_id=self.request_id,session_id=self.session_id,inference_index=self.inference_index)
            output=self.owner.infer(request);self.last_output=output;self.last_inference_s=now
            self.inference_index+=1;self.input_meta={k:request[k]for k in ['input_frame','input_time_s','history_frames','camera_frames','navigation','request_id','session_id','inference_index']}
            self.plan=None
            if output['valid']:
                xy=np.asarray(output['trajectory_xy_m']);local=np.c_[xy[:,0],-xy[:,1],np.zeros(len(xy))]
                self.plan=local@pose[:3,:3].T+pose[:3,3]
                self.plan_times=np.arange(1,len(xy)+1)*output['dt_s']
        diagnostic.update(status='predicted' if self.plan is not None else 'invalid_prediction',model_called=self.last_inference_s==now,
            inference_index=self.inference_index,input=self.input_meta,output=self.last_output)
        if self.plan is None:return dict(steer=0.,throttle=0.,brake=1.),diagnostic
        age=now-self.last_inference_s
        times=age+np.arange(1,9)*WAYPOINT_INTERVAL_S
        if times[-1]>self.plan_times[-1]+1e-6:raise ValueError('Cached model trajectory too old for its declared control horizon')
        world=np.column_stack([np.interp(times,self.plan_times,self.plan[:,j])for j in range(3)])
        local=(world-pose[:3,3])@pose[:3,:3]
        control, meta = self.tracker.step(local[:, [1, 0]], speed_mps)
        diagnostic.update(pid=meta,predicted_waypoints_forward_right_m=local[:,:2].tolist(),predicted_waypoint_spacing_s=WAYPOINT_INTERVAL_S,
            prediction_age_s=age,pose_vs_state_position_error_m=float(np.linalg.norm(pose[:3,3]-np.asarray(position_xyz))))
        return control,diagnostic

    def close(self):self.history=CausalHistory();self.plan=None
