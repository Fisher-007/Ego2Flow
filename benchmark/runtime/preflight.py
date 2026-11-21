"""Check the selected map and policy environments before simulation."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from benchmark import ROOT
from benchmark.runtime.execution import method_spec
from benchmark.runtime.server import server_paths


def check_map(resource, map_name):
    root = Path(resource['carla_root'])
    if not root.is_dir():
        raise FileNotFoundError(f"CARLA map installation: {root}")
    executable, project = server_paths(root, resource.get('ue4_root'))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"CARLA launch executable missing or not executable: {executable}")
    content = (project.parent if project else root / 'CarlaUE4') / 'Content'
    asset = map_name.removeprefix('/Game/') + '.umap'
    if (content / asset).is_file():
        return
    packages = [] if project else sorted((content / 'Paks').glob('*.pak'))
    if not packages:
        raise FileNotFoundError(content / asset)
    tool = shutil.which('UnrealPak')
    for location in (resource.get('ue4_root'), root):
        if location and os.access(Path(location) / 'Engine/Binaries/Linux/UnrealPak', os.X_OK):
            tool = str(Path(location) / 'Engine/Binaries/Linux/UnrealPak')
            break
    if not tool:
        raise RuntimeError(f'Cannot verify {map_name} inside the installed .pak archives. Add UnrealPak to PATH to inspect their contents.')
    for package in packages:
        listing = subprocess.run([tool, str(package), '-List'], check=True, capture_output=True, text=True).stdout
        mount = re.search(r'Mount point:?\s+([^\r\n]+)', listing, re.IGNORECASE)
        prefix = mount.group(1).strip().rstrip('/') if mount else ''
        entries = re.findall(r'"([^"\n]+\.umap)"', listing)
        if any(f'{prefix}/{entry}'.endswith('/Content/' + asset) or entry == asset for entry in entries):
            return
    raise FileNotFoundError(f'{map_name} in {content / "Paks"}')


def check_client(api, visual):
    sys.path.insert(0, api)
    import carla, numpy, scipy, shapely
    from agents.navigation.basic_agent import BasicAgent
    from benchmark.policies.adapters import policy_factory
    factory = visual.get('factory')
    if factory:
        policy_factory(factory)
    elif visual.get('kind') == 'lead':
        from benchmark.policies.lead import add_source_paths
        add_source_paths(visual['source_root'])
        from lead.inference.sensor_agent import SensorAgent
        (Path(visual['model_dir']) / 'config.json').resolve(strict=True)
    elif visual:
        raise ValueError(f"No adapter registered for {visual.get('kind')}")
    if factory == 'benchmark.policies.uniad:create':
        from mmcv.models.detectors.uniad_e2e import UniAD
        from benchmark.policies.uniad import stage2_settings
        stage2_settings(visual)


def check_environment(suite, resource, map_name):
    check_map(resource, map_name)
    env = {**os.environ, **resource.get('env', {}), 'CUDA_VISIBLE_DEVICES': str(resource['gpu'])}
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH'))))
    subprocess.run([resource['metrics_python'], '-c', 'import benchmark.metrics.evaluation'], cwd=ROOT, env=env, check=True)
    client_check = ('import json, sys; from benchmark.runtime.preflight import check_client; '
                    'check_client(sys.argv[1], json.loads(sys.argv[2]))')
    methods = [method_spec(suite, name) for name in suite['method_order']]
    clients = [(spec.get('python', resource['python']), spec.get('visual_policy', {})) for spec in methods]
    if all(python != resource['python'] for python, _ in clients):
        subprocess.run([resource['python'], '-c', 'import carla'], cwd=ROOT, env=env, check=True)
    for python, visual in clients:
        subprocess.run([python, '-c', client_check, resource['carla_python_api'], json.dumps(visual)],
                       cwd=ROOT, env=env, check=True)
        if visual.get('factory') != 'benchmark.policies.vlm:create':
            continue
        model_path = Path(visual['model_path'])
        if visual['model_kind'] == 'alpamayo':
            for relative in ('model/config.json', 'base-config/config.json', 'processor/preprocessor_config.json'):
                (model_path / relative).resolve(strict=True)
            imports = ('from alpamayo1_5 import helper; from alpamayo1_5.config import Alpamayo1_5Config; '
                       'from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5')
        else:
            (model_path / 'config.json').resolve(strict=True)
            imports = 'from transformers import AutoProcessor, Qwen3VLForConditionalGeneration'
        subprocess.run([visual['inference_python'], '-c', imports], cwd=ROOT, env=env, check=True)
