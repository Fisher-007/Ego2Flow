import unittest
from io import StringIO
from benchmark.navigation.native_controls import NativeControls


class NativeControlTests(unittest.TestCase):
    def test_static_stop_reference_is_found_without_signal_program(self):
        xml='''<OpenDRIVE><road id="1"><signals><signal id="s" s="5" type="206" dynamic="no"><validity fromLane="-1" toLane="-1"/></signal></signals></road>
        <road id="2"><signals><signalReference id="s" s="0.01"><validity fromLane="-1" toLane="-1"/></signalReference></signals></road></OpenDRIVE>'''
        points=[dict(road_id=2,lane_id=-1,s_m=s,position_carla_m=[s,0,0]) for s in (.5,2.5)]
        found=NativeControls(StringIO(xml)).on_route(points)
        self.assertEqual(len(found),1)
        self.assertEqual((found[0]['type'],found[0]['dynamic']),('206','no'))


if __name__=='__main__':unittest.main()
