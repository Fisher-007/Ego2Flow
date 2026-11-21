"""Resolve suite paths and runtime configuration."""
import gzip
import json
import os
from pathlib import Path
import sys
from benchmark import ROOT
from benchmark.runtime.execution import expand, read, write

METRICS = (
    'delay_s_per_request', 'throughput_veh_h', 'collision_rate_per_vehicle_km',
    'low_ttc_exposure_percent', 'hard_braking_rate_per_vehicle_km',
    'speed_sd_mps', 'amplification_ratio', 'estimated_energy_kwh_per_100_vehicle_km',
)


def defaults():
    os.environ['EGO2FLOW_ROOT'] = str(ROOT)
    os.environ.setdefault('SCENARIO_ROOT', str(ROOT / 'configs/scenes'))


def prepare(destination=None):
    defaults()
    destination = Path(destination or os.environ['SCENARIO_ROOT']).resolve()
    with gzip.open(ROOT / 'configs/six_cities_v1.json.gz', 'rt') as stream:
        scenes = json.load(stream)
    for scene_id, files in scenes.items():
        folder = destination / scene_id
        refresh = set()
        for filename, content in files.items():
            existing = folder / filename
            if not existing.is_file():
                continue
            installed = read(existing)
            if installed == content:
                continue
            if (filename == 'config.json' and isinstance(installed, dict) and 'measurement_route_aliases' not in content
                    and not (folder / 'measurements.json').exists()):
                installed.pop('measurement_route_aliases', None)
            if installed != content:
                raise ValueError(f'Existing scene differs: {existing}; choose a new destination')
            refresh.add(filename)
        for filename, content in files.items():
            target = folder / filename
            if filename in refresh or not target.exists():
                write(target, content)
    return destination


def path(value, base=ROOT):
    result = Path(expand(value)).expanduser()
    return str((base / result).resolve() if not result.is_absolute() else result)


def load_suite(filename, cities=None, methods=None, scenes=None):
    defaults()
    filename = Path(filename).resolve()
    # Resolve relative paths from the repository; expand variables and absolute paths.
    suite = expand(read(filename))
    selected = methods or suite['method_order']
    if len(selected) != len(set(selected)):
        raise ValueError('Select each method once')
    unknown = set(selected) - set(suite['methods'])
    if unknown: raise ValueError('Unknown methods: ' + ', '.join(sorted(unknown)))
    suite['methods'] = {m: path(suite['methods'][m]) for m in selected}
    suite['method_order'] = selected
    known = {a['city'] for a in suite['scenes']}
    all_ids = {a['id'] for a in suite['scenes']}
    if cities and set(cities) - known: raise ValueError('Unknown city selection')
    if scenes and set(scenes) - all_ids: raise ValueError('Unknown scene selection')
    suite['scenes'] = [a for a in suite['scenes'] if (not cities or a['city'] in cities)
                      and (not scenes or a['id'] in scenes)]
    ids = [a['id'] for a in suite['scenes']]
    if not ids or len(ids) != len(set(ids)): raise ValueError('Scene IDs must be unique and nonempty')
    if set(suite['metric_scenes']) != set(METRICS): raise ValueError('Declare all eight metric scene sets')
    for key, members in suite['metric_scenes'].items():
        if len(members) != len(set(members)): raise ValueError('Repeated scene in ' + key)
        if set(members) - all_ids: raise ValueError('Unknown applicable scene in ' + key)
        suite['metric_scenes'][key] = [name for name in members if name in ids]
    for scene in suite['scenes']:
        folder = Path(path(scene['scene_dir']))
        scene['scene_dir'] = str(folder)
        scene['config'] = path(scene.get('config', folder / 'config.json'))
        scene['deployment_configs'] = {d: path(p) for d,p in scene.get('deployment_configs', {}).items()}
    return suite


def resource_for_city(resource, city, output):
    defaults()
    resource = dict(resource)
    os.environ.update(expand(resource.get('env', {})))
    resource = expand(resource)
    resource['city'] = city
    resource['python'] = path(resource.get('python', sys.executable))
    resource['metrics_python'] = path(resource.get('metrics_python', resource['python']))
    resource['carla_root'] = path(resource['carla_root'])
    resource['carla_python_api'] = path(resource.get('carla_python_api',
                                                    resource['carla_root'] + '/PythonAPI/carla'))
    if resource.get('ue4_root'): resource['ue4_root'] = path(resource['ue4_root'])
    resource['tm_cache'] = str(Path(output).resolve() / 'cache' / city / 'native_tm.bin')
    resource['policy_logs'] = str(Path(output).resolve() / 'logs' / city / 'policies')
    return resource


def check_suite(suite, check_models=False):
    """Check declared inputs without loading a map or a model."""
    summaries = []
    maps = {}
    for scene in suite['scenes']:
        folder = Path(scene['scene_dir'])
        info = read(folder / 'map_info.json')
        if scene['city'] in maps and maps[scene['city']] != info['carla_map_name']:
            raise ValueError('A city worker requires one native map: ' + scene['city'])
        maps[scene['city']] = info['carla_map_name']
        routes = read(folder / 'candidate_routes.json')
        index = read(folder / 'scene_index.json')
        read(folder / 'signal_programs.json')
        if not any(a['scene_id'] == scene['id'] for a in index): raise ValueError('Scene index mismatch: '+scene['id'])
        cfg = read(scene['config'])
        if cfg['scene_id'] != scene['id'] or cfg['evaluation']['window_s'] != scene['window_s']:
            raise ValueError('Scene identity/window mismatch: ' + scene['id'])
        if not cfg.get('metric_scope'): raise ValueError('Missing metric scope: '+scene['id'])
        requests = cfg['request_manifest']
        if len({r['id'] for r in requests}) != len(requests): raise ValueError('Duplicate request identity')
        if any(r['route'] not in routes for r in requests): raise ValueError('Unknown request route: '+scene['id'])
        summaries.append(dict(scene=scene['id'],city=scene['city'],map=info['carla_map_name'],requests=len(requests)))
    if check_models:
        from benchmark.runtime.execution import method_spec
        for name in suite['method_order']:
            spec = method_spec(suite, name)
            visual = spec.get('visual_policy', {})
            if spec.get('python') and not Path(spec['python']).is_file():
                raise FileNotFoundError(f"{name}: interpreter: {spec['python']}")
            for key in ('source_root','model_dir','model_path','config','checkpoint','pid_source','calibration','inference_python'):
                if not visual.get(key): continue
                candidate = Path(visual[key])
                if key == 'checkpoint' and visual.get('model_dir'): candidate=Path(visual['model_dir']) / candidate
                if not candidate.exists(): raise FileNotFoundError(f'{name}: {key}: {candidate}')
            if visual.get('motion_anchor') and visual.get('factory') == 'benchmark.policies.uniad:create':
                (Path(visual['config']).parent / visual['motion_anchor']).resolve(strict=True)
    return summaries
