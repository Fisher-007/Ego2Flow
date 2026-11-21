from types import SimpleNamespace as S
import unittest
from shapely.geometry import box
from benchmark.scenes.entry_geometry import buffered_spawn_polygon


class EntryClearanceTests(unittest.TestCase):
    def test_parallel_entries_do_not_wait_for_an_unrelated_lane(self):
        pose=S(location=S(x=0,y=0),rotation=S(yaw=0))
        body=S(location=S(x=0,y=0),extent=S(x=2.5,y=1.))
        spawn=buffered_spawn_polygon(pose,body,2.)
        self.assertFalse(spawn.intersects(box(-2.5,2.2,2.5,4.2)))
        self.assertTrue(spawn.intersects(box(3.5,-1,8.5,1)))
        joint=buffered_spawn_polygon(pose,body,.1)
        self.assertFalse(joint.intersects(box(3.5,-1,8.5,1)))
