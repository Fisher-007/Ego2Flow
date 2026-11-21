"""Summarize traffic requests and vehicle controls for a run."""

import argparse
import json
from pathlib import Path
from benchmark.metrics.road_validity import road_sample_valid


def summarize(requests, frames, collisions, end_time):
    scheduled = requests
    # Limit request accounting to the elapsed observation window.
    requests = [r for r in requests if r["arrival_s"] < end_time]
    completed = [r for r in requests if r.get("completed_s") is not None]
    admitted = [r for r in requests if r.get("admitted_s") is not None]
    waiting = [r for r in requests if r.get("admitted_s") is None]
    active = [r for r in admitted if r.get("completed_s") is None and r.get('removed_s') is None]
    removed = [r for r in admitted if r.get('completed_s') is None and r.get('removed_s') is not None]
    system_time = sum((r.get("completed_s") if r.get("completed_s") is not None else end_time)
                      - r["arrival_s"] for r in requests)
    entry_time = sum((r.get("admitted_s") if r.get("admitted_s") is not None else end_time)
                     - r["arrival_s"] for r in requests)
    raw_frames = frames
    physical_vehicle_time = sum(len(row['vehicles']) * row['dt_s'] for row in frames)
    frames = [{**row, 'vehicles': [v for v in row['vehicles'] if road_sample_valid(v)]} for row in frames]
    vehicle_time = sum(len(row["vehicles"]) * row["dt_s"] for row in frames)
    speed_integral = sum(v["speed_mps"] * row["dt_s"] for row in frames for v in row["vehicles"])
    previous, distance = {}, 0.0
    far_runs, longest_far, far_distance = {}, 0.0, 0.0
    import math
    for row in frames:
        for vehicle in row["vehicles"]:
            key, position = vehicle["request_id"], vehicle["position_m"]
            old = previous.get(key)
            if old is not None:
                moved = math.dist(position[:2], old["position_m"][:2])
                distance += moved
                far = (vehicle.get("distance_from_probe_m") is not None
                       and old.get("distance_from_probe_m") is not None
                       and min(vehicle["distance_from_probe_m"], old["distance_from_probe_m"]) > 50)
                far_runs[key] = far_runs.get(key, 0) + row["dt_s"] if far and moved > 0.01 else 0
                if far and moved > 0.01:
                    far_distance += moved
                longest_far = max(longest_far, far_runs[key])
        previous = {v['request_id']: v for v in row['vehicles']}
    episodes, last_contact = [], {}
    for event in sorted(collisions, key=lambda e: e["time_s"]):
        pair = tuple(sorted((event["actor_id"], event["other_actor_id"])))
        if event["time_s"] - last_contact.get(pair, -float("inf")) > 1.0:
            episodes.append(event)
        last_contact[pair] = event["time_s"]
    stopped = lambda row: sum(v["speed_mps"] < 0.1 and (v.get("initial_prepared", False) or v.get("seconds_since_admission", 1) >= 1)
                              for v in row["vehicles"])
    queue_rows = [{"time_s": row["time_s"], "waiting_outside": len(row.get("pending_requests", [])),
                   "inside": len(row["vehicles"]), "stopped_inside": stopped(row)} for row in frames]
    result = {
        "scope": "All supplied routes and external entry queues over the observed window; not capacity or network MFD",
        "observation_seconds": end_time, "scheduled_requests": len(scheduled),
        "not_yet_due_at_cutoff": len(scheduled) - len(requests),
        "requests": len(requests), "admitted": len(admitted),
        "completed": len(completed), "waiting_outside": len(waiting), "unfinished_inside": len(active),
        "removed_unserved": len(removed),
        "request_accounting_balances": len(requests) == len(completed) + len(waiting) + len(active) + len(removed),
        "terminal_backlog": len(waiting) + len(active) + len(removed),
        "completion_fraction_of_due_requests": len(completed) / len(requests) if requests else None,
        "request_to_completion_or_cutoff_vehicle_seconds": system_time,
        "external_entry_wait_vehicle_seconds": entry_time,
        "mean_completed_request_time_s": (sum(r["completed_s"] - r["arrival_s"] for r in completed) / len(completed)) if completed else None,
        "completion_outflow_vehicles_per_hour": 3600 * len(completed) / end_time if end_time else None,
        "in_network_vehicle_seconds": physical_vehicle_time,
        "road_valid_vehicle_seconds": vehicle_time,
        "invalid_road_state_vehicle_seconds": physical_vehicle_time - vehicle_time,
        "stopped_in_network_vehicle_seconds": sum(stopped(row) * row["dt_s"] for row in frames),
        "queue_speed_threshold_mps": 0.1,
        "max_waiting_outside": max((row["waiting_outside"] for row in queue_rows), default=len(waiting)),
        "max_stopped_inside": max((row["stopped_inside"] for row in queue_rows), default=0),
        "terminal_stopped_inside": queue_rows[-1]["stopped_inside"] if queue_rows else 0,
        "queue_measurement": "Prepared initial vehicles count from observation time zero. Other entries exclude their first settling second; not a spatial queue length or observed traffic demand.",
        "time_weighted_mean_speed_mps": speed_integral / vehicle_time if vehicle_time else None,
        "tracked_distance_m": distance, "collision_episodes": len(episodes),
        "collision_episode_records": episodes,
        "longest_continuous_far_motion_s": longest_far, "far_vehicle_tracked_distance_m": far_distance,
        "unfinished_measurement": "Waiting and active requests contribute time through cutoff; completed-trip means exclude them and are reported separately.",
    }
    route_names = sorted({r["route"] for r in scheduled if "route" in r})
    if len(route_names) > 1:
        result["by_route"] = {}
        for name in route_names:
            selected = [r for r in scheduled if r["route"] == name]
            ids = {r["id"] for r in selected}
            route_frames = [{**row, "vehicles": [v for v in row["vehicles"] if v["request_id"] in ids],
                             "pending_requests": [i for i in row.get("pending_requests", []) if i in ids]}
                            for row in raw_frames]
            route_collisions = [e for e in collisions if e["request_id"] in ids]
            result["by_route"][name] = summarize(selected, route_frames, route_collisions, end_time)
        result["route_collision_counting"] = "An inter-route contact can appear in both routes; use the global episode count, not the sum of per-route counts."
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run / "run.json").read_text())
    requests = json.loads((args.run / "requests.json").read_text())
    frames = [json.loads(s) for s in (args.run / "frames.jsonl").read_text().splitlines()]
    collisions = [json.loads(s) for s in (args.run / "collisions.jsonl").read_text().splitlines()]
    result = summarize(requests, frames, collisions, metadata["elapsed_s"])
    (args.run / "run_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
