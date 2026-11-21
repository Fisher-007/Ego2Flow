import json
from pathlib import Path
import tempfile

from benchmark.runtime.collision_journal import CollisionJournal


def test_late_delivery_is_persisted_and_classified_by_occurrence_frame():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'raw.jsonl';journal=CollisionJournal(p)
        for frame in (99,102,101,104):
            journal.append(dict(frame=frame,actor_id=1,other_actor_id=2,request_id=0))
        assert len(p.read_text().splitlines())==4
        observed,outside=journal.select(100,102,.05,{1:0,2:1})
        assert [e['frame'] for e in observed]==[102,101]
        assert [round(e['time_s'],2) for e in observed]==[.15,.10]
        assert all(e['other_request_id']==1 for e in observed)
        assert [e['frame'] for e in outside]==[99,104]
        journal.close()


def test_quiet_before_shutdown_does_not_skip_waiting_for_pending_callback(tmp_path):
    import threading,time
    journal=CollisionJournal(tmp_path/'raw.jsonl')
    journal.last_delivery=time.monotonic()-10.
    pending=threading.Timer(.03,lambda:journal.append(dict(frame=102,actor_id=1,other_actor_id=2)))
    pending.start();result=journal.settle(quiet_s=.06,maximum_s=.5);pending.join()
    assert result['quiet_observed'] and result['wait_wall_s']>=.06
    observed,_=journal.select(100,102,.05,{1:0,2:1})
    assert len(observed)==1 and observed[0]['frame']==102
    journal.close()
