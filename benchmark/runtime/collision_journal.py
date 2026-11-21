"""Record and select CARLA collisions by the frame of impact."""
import json
import threading
import time


class CollisionJournal:
    def __init__(self, path):
        self.stream=path.open('w',buffering=1)
        self.lock=threading.Lock();self.events=[];self.last_delivery=time.monotonic()

    def append(self,event):
        with self.lock:
            value=dict(event,callback_received_monotonic_s=time.monotonic())
            self.stream.write(json.dumps(value)+'\n');self.stream.flush()
            self.events.append(value);self.last_delivery=time.monotonic()

    def settle(self, quiet_s=.5, maximum_s=3.):
        start=time.monotonic()
        while time.monotonic()-start<maximum_s:
            with self.lock:quiet=time.monotonic()-max(start,self.last_delivery)
            if quiet>=quiet_s:return dict(quiet_observed=True,wait_wall_s=time.monotonic()-start)
            time.sleep(.02)
        return dict(quiet_observed=False,wait_wall_s=time.monotonic()-start)

    def select(self, first_frame, last_frame, dt, actor_requests):
        with self.lock:events=list(self.events)
        observed=[];outside=[]
        for event in events:
            value=dict(event,time_s=(event['frame']-first_frame+1)*dt,
                       other_request_id=actor_requests.get(event['other_actor_id']))
            (observed if first_frame<=event['frame']<=last_frame else outside).append(value)
        return observed,outside

    def close(self):
        with self.lock:self.stream.close()
