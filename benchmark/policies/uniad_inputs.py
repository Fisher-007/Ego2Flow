"""Prepare UniAD camera, motion and navigation inputs from a CARLA frame."""
import json
import math
from pathlib import Path

import numpy as np


def route_target(route, position):
    """Advance the route and return the next point and command."""
    farthest, travelled, advance = -1.0, 0.0, 0
    for i in range(1, len(route)):
        if travelled > 50.0:
            break
        travelled += np.linalg.norm(route[i][0] - route[i - 1][0])
        distance = np.linalg.norm(route[i][0] - position[:2])
        if farthest < distance <= 4.0:
            farthest, advance = distance, i
    for _ in range(min(advance, len(route) - 2)):
        route.popleft()
    return route[min(1, len(route) - 1)]


def _camera_image(image):
    import cv2
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    _, jpeg = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 20])
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)


class UniADRig:
    def __init__(self, filename):
        rig = json.loads(Path(filename).read_text())
        cameras = rig['cameras']
        self.camera_ids = [c['id'] for c in cameras]
        self.lidar2cam = np.array([c['lidar2cam'] for c in cameras], dtype=float)
        self.lidar2img = np.array([c['lidar2img'] for c in cameras], dtype=float)
        self.lidar2ego = np.array(rig['lidar2ego'], dtype=float)
        self.imu_x_m = rig['imu_x_m']
        width, height = rig['image_size']
        self.sensors = []
        for c in cameras:
            x, y, z = c['position']
            self.sensors.append(dict(type='sensor.camera.rgb', id=c['id'], x=x, y=y, z=z,
                                     roll=0.0, pitch=0.0, yaw=c['yaw'], width=width, height=height,
                                     fov=c['fov'], sensor_tick=0))
        self.sensors.append(dict(type='sensor.other.imu', id='IMU', x=self.imu_x_m, y=0.0, z=0.0,
                                 roll=0.0, pitch=0.0, yaw=0.0, sensor_tick=0))

    def frame(self, inputs, speed, position, yaw_deg, target, command, step, box_type):
        from pyquaternion import Quaternion

        cameras = [_camera_image(inputs[name]) for name in self.camera_ids]
        image_shape = (*cameras[0].shape, len(cameras))
        imu = inputs['IMU']
        acceleration = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        compass = float(imu.compass)
        if math.isnan(compass):
            compass, acceleration, gyro = 0.0, np.zeros(3), np.zeros(3)

        # UniAD uses a right-handed world frame and [right, forward] planning points.
        heading = np.pi / 2 - compass
        rotation = Quaternion(axis=[0, 0, 1], radians=heading)
        can_bus = np.zeros(18)
        can_bus[:2] = position[0], -position[1]
        can_bus[3:7] = list(rotation)
        can_bus[7] = speed
        can_bus[10:13] = acceleration * [1, -1, 1]
        can_bus[13:16] = -gyro
        can_bus[16:] = heading, heading / np.pi * 180
        ego_pose = np.eye(4)
        ego_pose[:3, :3] = rotation.rotation_matrix
        ego_pose[:2, 3] = can_bus[:2]
        lidar_pose = ego_pose @ self.lidar2ego

        sample = dict(img=cameras, img_shape=image_shape, ori_shape=image_shape, pad_shape=image_shape,
                      lidar2img=self.lidar2img.copy(), lidar2cam=self.lidar2cam.copy(),
                      l2g_r_mat=lidar_pose[:3, :3], l2g_t=lidar_pose[:3, 3], can_bus=can_bus,
                      command=(4 if command < 0 else command) - 1, timestamp=step / 20.0,
                      scene_token=' ', folder=' ', frame_idx=0, box_type_3d=box_type)
        delta = target - position[:2]
        c, s = np.cos(compass), np.sin(compass)
        pid_target = np.array([[c, -s], [s, c]]) @ (delta * [1, -1])
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        local_target = [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
        heading_error = (compass - yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
        diagnostic = dict(target_forward_right_m=local_target, target_world_xy_m=target.tolist(),
                          command=command, imu_frame=imu.frame, imu_compass_rad=compass,
                          imu_compass_minus_pose_heading_rad=heading_error,
                          imu_acceleration_mps2=acceleration.tolist(), imu_gyroscope=gyro.tolist(),
                          localization_world_xyz_m=position.tolist())
        return sample, pid_target, diagnostic
