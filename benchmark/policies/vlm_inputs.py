"""Camera and pose inputs for OneVL and Alpamayo."""
from collections import deque
import math
import numpy as np

# CARLA's y axis points right and the model's points left.
FLIP=np.diag([1.,-1.,1.])
CV_FROM_CARLA=np.array([[0.,1.,0.],[0.,0.,-1.],[1.,0.,0.]])
ONEVL_CAMERAS=('ring_front_center','ring_front_right','ring_front_left')
ALPA_CAMERAS=('ring_front_left','ring_front_center','ring_front_right')


class CalibratedRig:
    def __init__(self,calibration):
        self.sensors=[];self.mounts={};self.remaps={}
        for name in ONEVL_CAMERAS:
            c=calibration[name];k=np.asarray(c['K']);fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
            focal=max(fx,fy);width=2*math.ceil(max(cx,c['width']-1-cx)*focal/fx+2)
            height=2*math.ceil(max(cy,c['height']-1-cy)*focal/fy+2)
            source=np.asarray(c['ego_SE3_camera_cv']);mount=np.eye(4)
            mount[:3,:3]=FLIP@source[:3,:3]@CV_FROM_CARLA;mount[:3,3]=FLIP@source[:3,3]
            rotation=mount[:3,:3];pitch=math.asin(float(np.clip(rotation[2,0],-1,1)))
            yaw=math.atan2(rotation[1,0],rotation[0,0]);roll=math.atan2(-rotation[2,1],rotation[2,2])
            self.mounts[name]=mount
            u,v=np.meshgrid(np.arange(c['width']),np.arange(c['height']))
            self.remaps[name]=(((u-cx)*focal/fx+width/2).astype(np.float32),((v-cy)*focal/fy+height/2).astype(np.float32))
            self.sensors.append(dict(id=name,type='sensor.camera.rgb',x=mount[0,3],y=mount[1,3],z=mount[2,3],
                pitch=math.degrees(pitch),yaw=math.degrees(yaw),roll=math.degrees(roll),width=width,height=height,
                fov=math.degrees(2*math.atan(width/(2*focal))),sensor_tick=0.,blueprint_attributes=dict(
                lens_k=0,lens_kcube=0,enable_postprocess_effects='True',motion_blur_intensity=0,
                lens_flare_intensity=0,bloom_intensity=0,chromatic_aberration_intensity=0)))

    def image(self,name,value):
        import cv2
        bgra=np.frombuffer(value.raw_data,np.uint8).reshape(value.height,value.width,4)
        return cv2.remap(bgra[:,:,:3],*self.remaps[name],interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()

    def pose(self,inputs):
        name=ONEVL_CAMERAS[0];sensor=inputs[name]
        return np.asarray(sensor.transform.get_matrix())@np.linalg.inv(self.mounts[name])


class CausalHistory:
    def __init__(self):self.poses=deque();self.images=deque();self.frame=None

    def append(self,frame,time_s,pose,images):
        if frame==self.frame:return
        self.frame=frame;self.poses.append((time_s,frame,np.asarray(pose)))
        self.images.append((time_s,frame,images))
        while self.poses and time_s-self.poses[0][0]>3.:self.poses.popleft()
        # Alpamayo uses the most recent 0.3 s of camera frames.
        while self.images and time_s-self.images[0][0]>.4:self.images.popleft()

    @staticmethod
    def required_seconds(kind):
        return 2.5 if kind=='onevl' else 1.5

    def ready(self,kind):
        return bool(self.poses) and self.poses[-1][0]-self.poses[0][0]>=self.required_seconds(kind)-1e-6

    def pose_at(self,time_s):
        row=min(self.poses,key=lambda p:abs(p[0]-time_s))
        if abs(row[0]-time_s)>.026:raise ValueError('Missing actual ego-history frame')
        return row

    def inputs(self,kind,rig):
        now,frame,origin=self.poses[-1]
        result=dict(input_frame=frame,input_time_s=now,origin_carla=origin.tolist())
        if kind=='onevl':
            states=[];frames=[]
            for offset in [-1.5,-1.,-.5,0.]:
                samples=[self.pose_at(now+offset-delay)for delay in [0.,.5,1.]]
                positions=[FLIP@origin[:3,:3].T@(p[2][:3,3]-origin[:3,3])for p in samples]
                a,b,c=positions;dt1=samples[0][0]-samples[1][0];dt0=samples[1][0]-samples[2][0]
                v1,v0=(a-b)/dt1,(b-c)/dt0
                states.append(dict(offset_s=offset,xy_m=a[:2].tolist(),velocity_xy_mps=v1[:2].tolist(),acceleration_xy_mps2=((v1-v0)/((dt1+dt0)/2))[:2].tolist()))
                frames.append([p[1]for p in samples])
            images=[rig.image(name,self.images[-1][2][name])for name in ONEVL_CAMERAS]
            result.update(states=states,history_frames=frames,camera_frames=[frame]*3,camera_order=list(ONEVL_CAMERAS))
        else:
            samples=[self.pose_at(now+i*.1)for i in range(-15,1)]
            xyz=[FLIP@origin[:3,:3].T@(p[2][:3,3]-origin[:3,3])for p in samples]
            rot=[FLIP@origin[:3,:3].T@p[2][:3,:3]@FLIP for p in samples]
            camera_rows=[min(self.images,key=lambda p:abs(p[0]-(now+i*.1))) for i in range(-3,1)]
            if any(abs(p[0]-(now+i*.1))>.026 for i,p in zip(range(-3,1),camera_rows)):raise ValueError('Missing actual camera history')
            images=[rig.image(name,row[2][name])for name in ALPA_CAMERAS for row in camera_rows]
            result.update(ego_history_xyz=np.asarray(xyz).tolist(),ego_history_rot=np.asarray(rot).tolist(),history_frames=[p[1]for p in samples],
                camera_frames=[p[1]for name in ALPA_CAMERAS for p in camera_rows],camera_order=list(ALPA_CAMERAS))
        # Plain bytes/lists avoid NumPy pickle incompatibility across py3.10/3.12.
        result['images']=[dict(shape=list(x.shape),data=x.tobytes())for x in images]
        return result


def motion_text(states):
    fields=[]
    for s in states:
        x,y=s['xy_m'];vx,vy=s['velocity_xy_mps'];ax,ay=s['acceleration_xy_mps2']
        fields.append(f'(t-{abs(s["offset_s"]):.1f}s) [{x:.2f}, {y:.2f}], Acceleration: X {ax:.2f}, Y {ay:.2f} m/s^2, Velocity: X {vx:.2f}, Y {vy:.2f} m/s')
    return 'The data is presented in the format [x, y]:\n'+', '.join(fields)+'\n'


def onevl_motion_text(states, encoding='numeric_history'):
    """Format ego motion history for the OneVL prompt."""
    encoding = encoding.removesuffix('_v1')
    if encoding not in ('numeric_history', 'stationary_summary'):
        raise ValueError('Unknown OneVL motion encoding: '+encoding)
    fields=('xy_m', 'velocity_xy_mps', 'acceleration_xy_mps2')
    stationary=bool(states) and all(float(f'{value:.2f}')==0.
        for state in states for field in fields for value in state[field])
    if encoding=='stationary_summary' and stationary:
        span=abs(states[0]['offset_s']-states[-1]['offset_s'])
        return f'Past ego motion: stationary over the last {span:.1f} seconds.\n', 'stationary_summary'
    return motion_text(states), 'numeric_history'
