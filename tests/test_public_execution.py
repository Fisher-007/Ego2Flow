"""Exercise local map workers, resume and scoring."""
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import os
import shlex
import sys
import pytest
from benchmark import ROOT, cli
from benchmark.runtime import execution
from benchmark.runtime.execution import read, write


def test_released_scenes_and_signal_bindings():
    suite = read(ROOT / 'configs/suite.json')
    with gzip.open(ROOT / 'configs/six_cities_v1.json.gz', 'rt') as stream:
        scenes = json.load(stream)
    cities = {scene['city'] for scene in suite['scenes']}
    tasks = ('S01', 'S02', 'S03', 'S04', 'P01', 'P02', 'B01', 'B02')
    assert set(scenes) == {scene['id'] for scene in suite['scenes']} == {
        f'{city}-{task}' for city in cities for task in tasks}
    for name, scene in scenes.items():
        assert set(scene) == {'config.json', 'map_info.json', 'scene_index.json',
                              'candidate_routes.json', 'signal_programs.json'}
        assert scene['config.json']['scene_id'] == name
        assert name in {entry['scene_id'] for entry in scene['scene_index.json']}
        assert 'initial_distribution_source' not in scene['config.json']
        assert 'measurement_route_aliases' not in scene['config.json']
        signals = scene['signal_programs.json']
        assert set(signals) == {'source_programs', 'bindings'}
        programs = {program['id']: program['phases'] for program in signals['source_programs']}
        for binding in signals['bindings']:
            phases = programs[binding['source_signal_id']]
            assert len(binding['phase_states']) == len(phases) > 0


def test_prepare_reuses_matching_files_and_preserves_edits(tmp_path, monkeypatch):
    from benchmark import configuration
    files = {'config.json': {'scene_id': 'ATX-B01'}, 'scene_index.json': [{'scene_id': 'ATX-B01'}],
             'map_info.json': {'map_version': 'v1.0'}}
    bundle = tmp_path / 'configs/six_cities_v1.json.gz'
    bundle.parent.mkdir()
    with gzip.open(bundle, 'wt') as stream:
        json.dump({'ATX-B01': files}, stream)
    monkeypatch.setattr(configuration, 'ROOT', tmp_path)
    monkeypatch.setenv('EGO2FLOW_ROOT', str(tmp_path))
    monkeypatch.setenv('SCENARIO_ROOT', str(tmp_path / 'configs/scenes'))
    root = tmp_path / 'configs/scenes'
    scene = root / 'ATX-B01'
    map_info = scene / 'map_info.json'
    write(scene / 'config.json', {**files['config.json'], 'measurement_route_aliases': {'lane0': 'through'}})
    write(map_info, {'map_version': 'custom'})
    with pytest.raises(ValueError, match='Existing scene differs'):
        configuration.prepare()
    assert read(map_info)['map_version'] == 'custom'
    assert not (scene / 'scene_index.json').exists()
    write(map_info, files['map_info.json'])
    write(scene / 'measurements.json', {})
    with pytest.raises(ValueError, match='Existing scene differs'):
        configuration.prepare()
    (scene / 'measurements.json').unlink()
    configuration.prepare()
    timestamps = {p: p.stat().st_mtime_ns for p in scene.iterdir()}
    configuration.prepare()
    assert {p: p.stat().st_mtime_ns for p in scene.iterdir()} == timestamps
    assert {p.name: read(p) for p in scene.iterdir()} == files


def test_parallel_resources_assign_exactly_one_map_per_device():
    runtime = read(ROOT / 'configs/runtime.parallel.example.json')
    cities = ['PAO', 'DTW', 'ATX', 'MIA', 'PIT', 'WDC']
    workers = cli.selected_resources(runtime, 'parallel', cities)
    assert [w['gpu'] for w in workers] == list(range(6))
    assert all(w['city'] in w['carla_root'] for w in workers)
    assert all(w['carla_python_api'] == runtime['resource']['carla_python_api'] for w in workers)
    first, second = runtime['workers'][:2]
    second['port'] = first['port'] + 10
    with pytest.raises(ValueError, match='port'):
        cli.selected_resources(runtime, 'parallel', cities)
    second['gpu'] = first['gpu']
    with pytest.raises(ValueError, match='GPU'):
        cli.selected_resources(runtime, 'parallel', cities)


def test_local_launch_passes_cpu_allocation_scene_and_output(tmp_path, monkeypatch):
    commands = []
    process = SimpleNamespace()
    def popen(command, **kwargs):
        assert kwargs['start_new_session']
        commands.append(command)
        return process
    monkeypatch.setattr(cli.subprocess, 'Popen', popen)
    resource = dict(city='PAO', gpu=1, cpus='8-15', port=23020)
    args = SimpleNamespace(output=tmp_path / 'city results', suite=ROOT / 'configs/suite.json',
                           technical_retries=1, methods=['IDM'], scenes=['PAO-S01'], command='run')
    assert cli.launch(resource, args) is process
    command, = commands
    assert command[:7] == ['taskset', '-c', '8-15', sys.executable, '-m', 'benchmark', '_map']
    assert command[command.index('--output') + 1] == str(args.output)
    assert command[command.index('--scenes') + 1] == 'PAO-S01'
    assert json.loads(command[command.index('--resource-json') + 1]) == resource
    assert (args.output / 'logs/PAO_launcher.log').is_file()


def test_sequential_resources_apply_each_map_before_launch():
    runtime = read(ROOT / 'configs/runtime.sequential.example.json')
    runtime['resource']['env'] = {'DISPLAY': ':0'}
    runtime['resource']['maps']['DTW'].update(cpus='8-15', env={'DISPLAY': ':1'})
    pao, dtw = cli.selected_resources(runtime, 'sequential', ['PAO', 'DTW'])
    assert pao['gpu'] == dtw['gpu'] and pao['port'] == dtw['port']
    assert pao['env']['DISPLAY'] == ':0' and dtw['env']['DISPLAY'] == ':1'
    assert 'DTW' in dtw['carla_root'] and dtw['cpus'] == '8-15'


def test_server_launcher_uses_the_city_log_directory(tmp_path, monkeypatch):
    server_dir = tmp_path / 'carla_servers' / 'PAO'
    old_ready = tmp_path / 'worlds/PAO/ready.json'
    write(old_ready, {'map': 'old'})
    process = SimpleNamespace()

    def launch(command, **kwargs):
        assert Path(command[command.index('--output') + 1]) == server_dir
        assert Path(kwargs['stdout'].name) == server_dir / 'launcher.log'
        write(server_dir / 'ready.json', {'map': 'PAO'})
        return process

    monkeypatch.setattr(execution.subprocess, 'Popen', launch)
    resource = dict(city='PAO', cpus='0', python=sys.executable, port=23000)
    assert execution.start_world({'scene_dir': 'PAO-S01'}, resource, tmp_path, {}) is process
    assert read(old_ready) == {'map': 'old'}


def test_runtime_selects_local_parallel_mode_when_mode_is_omitted(tmp_path, monkeypatch):
    launches = []
    def launch(resource, args):
        launches.append((resource['city'], resource['gpu']))
        return SimpleNamespace(wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(cli, 'launch', launch)
    monkeypatch.setattr(cli, 'check_suite', lambda suite: None)
    runtime = ROOT / 'configs/runtime.parallel.example.json'
    assert cli.main(['doctor', '--runtime', str(runtime), '--methods', 'IDM',
                     '--cities', 'PAO', 'MIA', '--output', str(tmp_path)]) == 0
    assert sorted(launches) == [('MIA', 3), ('PAO', 0)]
    assert not (tmp_path / 'suite.json').exists()


@pytest.mark.parametrize('mode, active', [('sequential', ['PAO']), ('parallel', ['PAO', 'MIA'])])
def test_ctrl_c_waits_for_active_cities(tmp_path, monkeypatch, mode, active):
    started, stopping, closed = [], [], []

    class MapProcess:
        def __init__(self, resource, args):
            self.city = resource['city']
            started.append(self.city)

        def poll(self):
            return None

        def terminate(self):
            stopping.append(self.city)

        def wait(self):
            if self.city not in stopping:
                raise KeyboardInterrupt
            closed.append(self.city)
            return 130

    monkeypatch.setattr(cli, 'launch', MapProcess)
    monkeypatch.setattr(cli, 'check_suite', lambda suite: None)
    runtime = ROOT / f'configs/runtime.{mode}.example.json'
    assert cli.main(['run', '--runtime', str(runtime), '--methods', 'IDM',
                     '--cities', 'PAO', 'MIA', '--output', str(tmp_path)]) == 130
    assert started == stopping == closed == active


def test_doctor_checks_selected_map_launcher_and_method_interpreter(tmp_path):
    from benchmark.configuration import METRICS
    folder = tmp_path / 'scene'
    native = tmp_path / 'native'
    project = native / 'Unreal/CarlaUE4/CarlaUE4.uproject'
    engine = tmp_path / 'engine/Engine/Binaries/Linux/UE4Editor'
    asset = project.parent / 'Content/City/Maps/City.umap'
    method = tmp_path / 'method.json'
    interpreter = tmp_path / 'method-python'
    interpreter.write_text('#!/bin/sh\nexport METHOD_ENV=active\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
    interpreter.chmod(0o755)
    api = tmp_path / 'api'
    modules = tmp_path / 'modules'
    modules.mkdir()
    (modules/'carla.py').touch()
    for name in ('agents/__init__.py', 'agents/navigation/__init__.py', 'agents/navigation/basic_agent.py'):
        path = api/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('class BasicAgent: pass\n' if path.stem == 'basic_agent' else '')
    write(folder/'map_info.json', {'carla_map_name': '/Game/City/Maps/City'})
    write(folder/'scene_index.json', [{'scene_id': 'PAO-S01'}])
    write(folder/'candidate_routes.json', {})
    write(folder/'signal_programs.json', {})
    write(folder/'config.json', dict(scene_id='PAO-S01', evaluation=dict(window_s=[0, 2]),
                                    metric_scope={'window_s': [0, 2]}, request_manifest=[]))
    write(method, dict(id='Example', python=str(interpreter), visual_policy=dict(kind='custom', factory='my_policy:create')))
    suite = tmp_path/'suite.json'
    write(suite, dict(scenes=[dict(id='PAO-S01', city='PAO', scene_dir=str(folder), window_s=[0, 2])],
                      methods={'Example': str(method)}, method_order=['Example'], metric_scenes={k: [] for k in METRICS}))
    runtime = tmp_path/'runtime.json'
    write(runtime, dict(resource=dict(gpu=0, cpus=str(min(os.sched_getaffinity(0))), port=23000,
          carla_root=str(native), ue4_root=str(tmp_path/'engine'), carla_python_api=str(api), env={'PYTHONPATH': str(modules)})))
    out = tmp_path/'results'
    args = ['doctor', '--runtime', str(runtime), '--suite', str(suite), '--output', str(out)]
    log = out/'logs/PAO_launcher.log'
    assert cli.main(args) == 1 and str(native) in log.read_text()
    project.parent.mkdir(parents=True)
    project.touch()
    assert cli.main(args) == 1 and str(engine) in log.read_text()
    engine.parent.mkdir(parents=True)
    engine.touch(mode=0o755)
    assert cli.main(args) == 1 and str(asset) in log.read_text()
    asset.parent.mkdir(parents=True)
    asset.touch()
    assert cli.main(args) == 1 and 'No module named \'my_policy\'' in log.read_text()
    (modules/'my_policy.py').write_text("import os\nassert os.environ['METHOD_ENV'] == 'active'\n"
                                        "def create(config): raise AssertionError('loaded a model')\n")
    assert cli.main(args) == 0


def test_resume_distinguishes_paths_from_driving_inputs(tmp_path):
    import copy
    cfg = dict(method_version='1', rule_controller=dict(kind='traffic_manager', native_tm_cache='/old', path_spacing_m=20))
    original = dict(evaluator='test', config=cfg, scene_files={'candidate_routes.json': {'lane': 'one'}})
    write(tmp_path/'run.json', dict(execution_completed=True))
    write(tmp_path/'input_receipt.json', original)
    relocated = copy.deepcopy(original)
    relocated['config']['rule_controller']['native_tm_cache'] = '/new'
    assert execution.reusable(tmp_path, relocated)
    for change in [('method_version', '2'), ('rule_controller', {**cfg['rule_controller'], 'path_spacing_m': 10})]:
        assert not execution.reusable(tmp_path, {**relocated, 'config': {**relocated['config'], change[0]: change[1]}})
    assert not execution.reusable(tmp_path, {**relocated, 'scene_files': {'candidate_routes.json': {'lane': 'two'}}})

    visual = dict(factory='benchmark.policies.vlm:create', checkpoint='original', diagnostics_dir='/old')
    for factory, reuse in [(visual['factory'], True), ('my_policy:create', False)]:
        original['config'] = {'visual_policy': dict(visual, factory=factory)}
        write(tmp_path/'input_receipt.json', original)
        changed = copy.deepcopy(original)
        changed['config']['visual_policy']['diagnostics_dir'] = '/new'
        assert execution.reusable(tmp_path, changed) is reuse
        changed['config']['visual_policy']['checkpoint'] = 'different'
        assert not execution.reusable(tmp_path, changed)


def test_cooked_map_check_uses_the_requested_asset(tmp_path):
    from benchmark.runtime.preflight import check_map
    binary = tmp_path / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    content = tmp_path / 'CarlaUE4/Content'
    asset = content / 'First/Maps/First.umap'
    asset.parent.mkdir(parents=True)
    asset.touch()
    resource = {'carla_root': str(tmp_path)}
    check_map(resource, '/Game/First/Maps/First')
    with pytest.raises(FileNotFoundError):
        check_map(resource, '/Game/Second/Maps/Second')
    pak = content / 'Paks/maps.pak'
    pak.parent.mkdir()
    pak.touch()
    tool = tmp_path / 'Engine/Binaries/Linux/UnrealPak'
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\necho 'Mount point ../../../CarlaUE4/Content/Second/'\necho '\"Maps/Second.umap\"'\n")
    tool.chmod(0o755)
    check_map(resource, '/Game/Second/Maps/Second')
    with pytest.raises(FileNotFoundError):
        check_map(resource, '/Game/Third/Maps/Third')


def test_finite_retry_reuses_completed_condition_and_preserves_failed_attempt(tmp_path,monkeypatch):
    scene_dir=tmp_path/'PAO-S01'
    for name in ('map_info.json','scene_index.json','candidate_routes.json','signal_programs.json'):
        write(scene_dir/name,{})
    config=scene_dir/'config.json';write(config,dict(seed=7,scene_id='PAO-S01',duration_s=60.,
                                               evaluation=dict(window_s=[0.,60.]),request_manifest=[]))
    method=tmp_path/'method.json';write(method,dict(id='Example',name='Example',version='v1',rule_controller=dict(kind='idm')))
    suite=dict(methods={'Example':str(method)},method_order=['Example'],
               scenes=[dict(id='PAO-S01',city='PAO',version='v1',config=str(config),scene_dir=str(scene_dir))])
    resource=dict(city='PAO',cpus='0',gpu=0,port=23000,python=sys.executable,
                  carla_root='/carla',carla_python_api='/carla/PythonAPI/carla',tm_cache='/cache',policy_logs='/logs')
    starts=[];attempts=[]
    monkeypatch.setattr(execution,'start_world',lambda *args: starts.append(1) or None)
    monkeypatch.setattr(execution,'score',lambda *args:True)
    class Process:
        def __init__(self,command,**kwargs):
            batch=read(Path(command[command.index('--batch')+1]))
            self.returncode=0
            for case in batch['cases']:
                output=Path(case['output']);attempts.append(output.name)
                if output.name.endswith('_M') and attempts.count(output.name)==1:
                    write(output/'public_status.json',dict(status='technical_failure'))
                    self.returncode=1;break
                write(output/'run.json',dict(execution_completed=True))
                write(output/'public_status.json',dict(status='observed'))
        def poll(self):return self.returncode
    monkeypatch.setattr(execution.subprocess,'Popen',Process)
    output=tmp_path/'output'
    assert execution.worker(suite,resource,['Example'],output,retries=1)==0
    assert attempts==['PAO-S01_H','PAO-S01_M','PAO-S01_M']
    backups=list((output/'backup').glob('*/Example/runs/PAO-S01_M/public_status.json'))
    assert len(backups)==1 and read(backups[0])['status']=='technical_failure'
    run_h = output / 'Example/runs/PAO-S01_H'
    receipt_path = run_h / 'input_receipt.json'
    saved = read(receipt_path)
    saved['config']['initial_distribution_source'] = {'log_id': 'observation', 'timestamp_ns': 1}
    saved['config']['measurement_route_aliases'] = {'lane0': 'through'}
    write(receipt_path, saved)
    starts_before_resume = len(starts)
    assert execution.worker(suite,resource,['Example'],output,retries=1)==0
    assert len(attempts)==3  # No second simulator/model launch for unchanged complete inputs.
    assert len(starts) == starts_before_resume
    write(config,dict(read(config),duration_s=90.))
    assert execution.worker(suite,resource,['Example'],output,retries=1)==0
    assert attempts[3:] == ['PAO-S01_H', 'PAO-S01_M']
    assert (run_h / 'run.json').is_file()
    assert len(list((output/'backup').glob('*/Example/runs/PAO-S01_H/run.json'))) == 1


def test_rescore_and_report_entrypoint_from_complete_observations(tmp_path):
    import shutil
    from benchmark.configuration import METRICS
    from tests.test_evaluation_pipeline import fixture
    from benchmark.cli import main
    source=tmp_path/'source';source.mkdir()
    raw=fixture(source)
    write(raw/'run.json',dict(read(raw/'run.json'),execution_completed=True))
    output=tmp_path/'results'
    runs = output / 'Example/runs'
    for deployment in ('H', 'M'):
        target = runs / f'example_{deployment}'
        shutil.copytree(raw,target)
        write(target/'config.json',dict(read(target/'config.json'),deployment=deployment))
    write(runs/'other_H/run.json', dict(execution_completed=True))
    method=tmp_path/'example.json';write(method,dict(id='Example',name='Example',version='v1'))
    suite=tmp_path/'suite.json';write(suite,dict(
        method_order=['Example'],methods={'Example':str(method)},
        scenes=[dict(id='example',city='PAO',window_s=[0.,2.],scene_dir=str(source/'scene'))],
        metric_scenes={k:[] if k=='amplification_ratio' else ['example'] for k in METRICS}))
    assert main(['rescore','--suite',str(suite),'--output',str(output)])==0
    result=read(output/'report/results.json')
    assert (runs/'example_H/evaluation.json').is_file()
    assert (runs/'example_M/evaluation.json').is_file()
    assert not (runs/'other_H/evaluation.json').exists()
    assert len(result['conditions']) == 2
    assert result['main'][0]['observed']==2
    assert result['main'][0]['raw_metrics']['delay_s_per_request'] == pytest.approx(4 / 3)
    assert (output/'report/RESULTS.md').exists() and (output/'report/conditions.csv').exists()
    assert result['main'][0]['normalized_metrics']['amplification_ratio'] is None
    assert result['main'][0]['score'] is None  # A service-only subset has no total.
    route_path = source / 'scene/candidate_routes.json'
    routes = read(route_path)
    for point in routes['lane0']['validated_waypoints']:
        point['free_flow_speed_mps'] = 20
    write(route_path, routes)
    saved = runs / 'example_M/input_receipt.json'
    args = ['rescore', '--suite', str(suite), '--output', str(output)]
    assert main(args) == 0
    assert read(output/'report/results.json')['main'] == result['main']
    saved.unlink()
    previous = read(saved.parent/'evaluation.json')
    with pytest.raises(ValueError, match='no saved routes'):
        main(args)
    assert read(saved.parent/'evaluation.json') == previous
    assert main([*args, '--use-suite-scenes']) == 0
    assert read(saved.parent/'evaluation.json')['metrics']['delay_s_per_request']['value'] == pytest.approx(5 / 3)
    assert read(runs/'example_H/evaluation.json')['metrics']['delay_s_per_request']['value'] == pytest.approx(4 / 3)
