"""Launch an owned CARLA server for a configured benchmark map."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def server_paths(carla_root, ue4_root):
    project = carla_root / "Unreal/CarlaUE4/CarlaUE4.uproject"
    if project.is_file():
        if not ue4_root:
            raise ValueError("Set UE4_ROOT to the Unreal Engine build for the native map project")
        return Path(ue4_root) / "Engine/Binaries/Linux/UE4Editor", project
    return carla_root / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    args = parser.parse_args()
    import carla

    for name in ("CARLA_ROOT", "E2F_CPUSET", "E2F_GPU"):
        if not os.environ.get(name):
            parser.error(f"Set {name} to this experiment's allocated resource")
    port = int(os.environ.get("E2F_RPC_PORT", 23000))
    for candidate in (port, port + 1, port + 2):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", candidate))
    info = json.loads((args.scene_dir / "map_info.json").read_text())
    carla_root = Path(os.environ["CARLA_ROOT"]).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["E2F_GPU"], OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    runtime = Path(f"/tmp/e2f-benchmark-{port}")
    for category in ("CONFIG", "CACHE", "DATA"):
        location = runtime / category.lower()
        location.mkdir(parents=True, exist_ok=True)
        env[f"XDG_{category}_HOME"] = str(location)
    if env.get("E2F_VULKAN_LOADER"):
        env["LD_LIBRARY_PATH"] = env["E2F_VULKAN_LOADER"] + ":" + env.get("LD_LIBRARY_PATH", "")
    executable, project = server_paths(carla_root, env.get("UE4_ROOT"))
    if project:
        command = [str(executable), str(project), info["carla_map_name"], "-game",
                   "-ini:Engine:[DevOptions.Shaders]:bAllowAsynchronousShaderCompiling=False"]
    else:
        command = [str(executable), "CarlaUE4", info["carla_map_name"]]
    command += ["-RenderOffScreen", "-vulkan", "-graphicsadapter=" + env["E2F_GPU"],
                f"-carla-rpc-port={port}", "-quality-level=Epic", "-nosound", "-nocrashreports"]
    command += json.loads(env.get("E2F_SERVER_ARGUMENTS", "[]"))
    command = ["taskset", "-c", env["E2F_CPUSET"], *command]
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=carla_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started owned server PID {server.pid}, port {port}", flush=True)
        (output/'server_process.json').write_text(json.dumps(dict(server_pid=server.pid,launcher_pid=os.getpid(),port=port))+'\n')
        def finish(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,finish)
        try:
            started = time.monotonic()
            # Reuse one client while CARLA finishes loading after opening its RPC port.
            client = None
            while time.monotonic() - started < args.startup_timeout:
                if server.poll() is not None:
                    raise RuntimeError("CARLA exited during startup; inspect server.log")
                if client is None:
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            pass
                    except OSError:
                        time.sleep(2)
                        continue
                    client = carla.Client("127.0.0.1", port)
                    client.set_timeout(120)
                try:
                    client.get_world()
                    break
                except RuntimeError:
                    time.sleep(1)
            else:
                raise TimeoutError("CARLA startup exceeded the configured timeout")
            # Each OpenDRIVE request transfers the entire map.
            client.set_timeout(600)
            name = client.get_world().get_map().name
            if name.removeprefix("/Game/") != info["carla_map_name"].removeprefix("/Game/") and not project:
                # CARLA 0.9.16 can start on the default map despite the launch argument.
                print(f"Packaged server booted {name}; loading requested map", flush=True)
                client.set_timeout(180)
                name = client.load_world(info["carla_map_name"].rsplit("/", 1)[-1]).get_map().name
            if name.removeprefix("/Game/") != info["carla_map_name"].removeprefix("/Game/"):
                raise ValueError("Unexpected map loaded")
            print(f"Map ready after {time.monotonic() - started:.1f}s", flush=True)
            (output/'ready.json').write_text(json.dumps(dict(map=name,startup_wall_s=time.monotonic()-started))+'\n')
            while server.poll() is None:
                time.sleep(5)
            raise RuntimeError('Owned CARLA server exited')
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            print("Stopped this server process group", flush=True)


if __name__ == "__main__":
    main()
