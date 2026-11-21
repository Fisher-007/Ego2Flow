"""Build deterministic demand schedules with stable request identities."""

import math


def arrival_times(config):
    duration = config["duration_s"]
    if "demand" in config:
        spec = config["demand"]
        start, end = spec.get("start_s", 0), spec.get("end_s", duration)
        headway = spec["headway_s"]
        if not (0 <= start < end <= duration and headway > 0):
            raise ValueError("Demand needs positive headway and 0 <= start < end <= duration")
        values = [round(start + i * headway, 9)
                  for i in range(math.ceil((end - start) / headway))
                  if start + i * headway < end]
    else:
        values = list(config["arrival_times_s"])
    if values != sorted(values) or any(t < 0 or t >= duration for t in values):
        raise ValueError("Arrivals must be sorted inside the half-open observation window")
    return values


def scheduled_requests(config):
    """Merge entrance streams into stable request IDs, retaining route identity."""
    if "request_manifest" in config:
        return [{**r, "admitted_s": None, "completed_s": None, "spawn_attempts": 0}
                for r in sorted(config["request_manifest"], key=lambda r: (r["arrival_s"], r["id"]))]
    if "demand_streams" in config:
        streams = config["demand_streams"]
    else:
        streams = [{"route": config["route"],
                    **({"demand": config["demand"]} if "demand" in config
                       else {"arrival_times_s": config["arrival_times_s"]})}]
    scheduled = [{"route": item["route"], "entry": item.get("entry", "initial_"+str(i)), "arrival_s": 0.0,
                  "origin": "initial_population", "initial_speed_mps": item["initial_speed_mps"],
                  **{key: item[key] for key in ("vehicle_blueprint", "target_speed_kmh") if key in item}}
                 for i, item in enumerate(config.get("initial_vehicles", []))]
    for stream in streams:
        for arrival in arrival_times({"duration_s": config["duration_s"], **stream}):
            scheduled.append({"route": stream["route"],
                              "entry": stream.get("entry", stream["route"]), "arrival_s": arrival,
                              **{key: stream[key] for key in ("vehicle_blueprint", "target_speed_kmh") if key in stream}})
    scheduled.sort(key=lambda r: r["arrival_s"])
    return [{"id": i, **r, "admitted_s": None, "completed_s": None, "spawn_attempts": 0}
            for i, r in enumerate(scheduled)]


def entry_heads(pending):
    """One FIFO head per entrance: a blocked entrance cannot block another."""
    heads = {}
    for request in pending:
        heads.setdefault(request["entry"], request)
    return list(heads.values())
