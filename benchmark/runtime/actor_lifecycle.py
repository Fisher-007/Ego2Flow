"""Retain requests for actors removed by the simulator."""


def retire_missing_vehicles(active, snapshot, elapsed, emit):
    removed = []
    for key, item in list(active.items()):
        # Native TM can mark an actor PendingKill while it is still in the
        # current snapshot. Presence alone does not permit live actor RPCs.
        if snapshot.find(item['vehicle'].id) is not None and item['vehicle'].is_alive:
            continue
        request = item['request']
        request.update(removed_s=elapsed, removal_reason='simulator_actor_missing')
        request.setdefault('failure_reason', 'simulator_actor_missing')
        emit('simulator_actor_missing', request, actor_id=item['vehicle'].id,
             last_observed_state=item.get('last_observed_state'),
             request_retained_unserved=True)
        removed.append(active.pop(key))
    return removed
