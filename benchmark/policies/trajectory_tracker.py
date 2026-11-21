"""Track predicted waypoints with steering and speed PID control."""
from collections import deque

import numpy as np


WAYPOINT_INTERVAL_S = 0.5


class _WindowPID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = deque([0.0] * 40, maxlen=40)

    def step(self, error):
        change = error - self.errors[-1]
        self.errors.append(error)
        return self.kp * error + self.ki * np.mean(self.errors) + self.kd * change


def _steering_error(point):
    # Waypoints are [right, forward] in metres. Steering error is in quarter turns.
    right, forward = point
    return float(np.degrees(np.pi / 2 - np.arctan2(forward, right)) / 90)


class TrajectoryTracker:
    def __init__(self):
        self.steering = _WindowPID(kp=0.75, ki=0.75, kd=0.3)
        self.speed = _WindowPID(kp=5.0, ki=0.5, kd=1.0)

    def step(self, waypoints, speed_mps, navigation_target=None):
        waypoints = np.asarray(waypoints)
        segments = np.diff(waypoints, axis=0)
        duration_s = len(segments) * WAYPOINT_INTERVAL_S
        desired_speed = float(sum(np.linalg.norm(segment) / duration_s for segment in segments))

        midpoints = (waypoints[:-1] + waypoints[1:]) / 2
        distance_m = np.linalg.norm(midpoints, axis=1)
        aim = waypoints[np.argmin(np.abs(distance_m - 4.0))]
        angle = _steering_error(aim)
        angle_last = _steering_error(segments[-1])
        target = aim if navigation_target is None else np.asarray(navigation_target)
        angle_target = angle if navigation_target is None else _steering_error(target)
        angle_final = angle
        if navigation_target is not None:
            close_turn = target[1] < 10.0 and abs(angle_target - angle_last) > 0.3
            if abs(angle_target) < abs(angle) or close_turn:
                angle_final = angle_target
        steer = float(np.clip(self.steering.step(angle_final), -1.0, 1.0))

        speed_error = float(np.clip(desired_speed - speed_mps, 0.0, 0.25))
        throttle = float(np.clip(self.speed.step(speed_error), 0.0, 0.75))
        brake = desired_speed < 0.4 or speed_mps / desired_speed > 1.1
        control = dict(steer=steer, throttle=0.0 if brake else throttle, brake=float(brake))

        diagnostics = dict(
            control, speed=float(speed_mps), desired_speed=desired_speed, delta=speed_error,
            aim=aim.tolist(), target=target.tolist(), angle=angle, angle_target=angle_target,
            angle_final=angle_final, angle_last=angle_last,
        )
        for i, point in enumerate(waypoints[:4], start=1):
            diagnostics[f'wp_{i}'] = point.tolist()
        return control, diagnostics
