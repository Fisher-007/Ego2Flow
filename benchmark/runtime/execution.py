"""Run each driving method in order on one map."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from benchmark import ROOT
from benchmark.runtime.progress_log import progress

VERSION = 'ego2flow-public-v1'


def worker_name(resource):
    return resource['city']


def worker_cases(suite, methods, resource):
    return cases_for(suite, methods, resource['city'])

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def expand(value):
    if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
    if isinstance(value,list):return [expand(v) for v in value]
    if isinstance(value,str):
        result=os.path.expandvars(value)
        if '${' in result:raise ValueError('Unresolved environment variable: '+value)
        return result
    return value

def cases_for(suite,methods,city):
    return [dict(method=method,scene=scene,deployment=deployment)
            for method in methods for scene in suite['scenes'] if scene['city']==city
            and (method not in suite.get('method_scene_ids',{}) or scene['id'] in suite['method_scene_ids'][method])
            for deployment in ('H','M')]

def case_paths(root, case):
    name = f"{case['scene']['id']}_{case['deployment']}"
    return root / case['method'] / 'runs' / name

def method_spec(suite,method):
    spec=read(suite['methods'][method])
    if spec['id'] != method:
        raise ValueError('Method JSON ID differs from suite key: ' + method)
    visual = spec.get('visual_policy') or {}
    if visual.get('factory') == 'benchmark.policies.vlm:create':
        visual.pop('pid_source', None)
    if visual.get('factory') == 'benchmark.policies.uniad:create' and 'source_root' in visual:
        root = os.path.expandvars(visual.pop('source_root'))
        if '${' not in root:
            visual.setdefault('asset_root', root)
    spec.setdefault('status', 'ready')
    return expand(spec)

def configuration(scene,method,deployment,resource):
    source=scene.get('deployment_configs',{}).get(deployment,scene['config'])
    cfg=copy.deepcopy(read(source))
    cfg.pop('rule_controller',None);cfg.pop('visual_policy',None)
    cfg.pop('initial_distribution_source', None)
    cfg.update(cfg.pop('adapter_preparation', {}).get(method['id'], {}).get(deployment, {}))
    for key in ('rule_controller','visual_policy'):
        if key in method:cfg[key]=copy.deepcopy(method[key])
    if cfg.get('visual_policy'):
        cfg['visual_policy'].setdefault('seed', cfg['seed'])
    cfg.update(controller=method['name'],deployment=deployment,reuse_empty_world=True,
        fresh_server_process=False,reload_world_before_run=False,
        public_evaluator_version=VERSION,method_version=method['version'],scene_version=scene['version'])
    if cfg.get('rule_controller',{}).get('kind')=='traffic_manager':
        cfg['rule_controller']['native_tm_cache']=resource['tm_cache']
    if cfg.get('visual_policy', {}).get('kind') in ('onevl', 'alpamayo'):
        cfg['visual_policy']['diagnostics_dir']=resource['policy_logs'] + '/' + method['id']
    return cfg

def receipt(scene,cfg):
    scene_dir=Path(scene['scene_dir'])
    files={name:read(scene_dir/name) for name in
           ('map_info.json','scene_index.json','candidate_routes.json','signal_programs.json')}
    measurements=scene_dir/'measurements.json'
    if measurements.is_file():
        files['measurements.json']=read(measurements)
    return dict(evaluator=VERSION,config=cfg,scene_files=files)


def resume_config(config, scene_files):
    cfg = dict(config)
    cfg.pop('initial_distribution_source', None)
    if 'measurements.json' not in scene_files:
        cfg.pop('measurement_route_aliases', None)
    rule = cfg.get('rule_controller', {})
    if rule.get('kind') == 'traffic_manager' and 'native_tm_cache' in rule:
        cfg['rule_controller'] = {**rule, 'native_tm_cache': bool(rule['native_tm_cache'])}
    visual = cfg.get('visual_policy', {})
    if visual.get('factory') == 'benchmark.policies.vlm:create':
        cfg['visual_policy'] = {k: v for k, v in visual.items() if k != 'diagnostics_dir'}
    return cfg

def reusable(output,expected):
    run = output / 'run.json'
    receipt_file = output / 'input_receipt.json'
    if not receipt_file.is_file() or not run.is_file() or not read(run).get('execution_completed'):
        return False
    saved = read(receipt_file)
    if 'config' not in saved:
        return False
    saved['config'] = resume_config(saved['config'], saved.get('scene_files', {}))
    return saved == {**expected, 'config': resume_config(expected['config'], expected['scene_files'])}

def archive_condition(output,root):
    """Keep replaced attempts outside the active result tree."""
    root=Path(root);output=Path(output)
    archive=root/'backup'/('replaced_attempt_'+str(time.time_ns()))
    target=archive/output.relative_to(root)
    target.parent.mkdir(parents=True,exist_ok=True)
    output.rename(target)
    (archive/'README.md').write_text(
        '# Replaced attempt\n\n'
        'Original active location: `'+str(output)+'`.\n\n'
        'Reason: unfinished run or input receipt differs from the requested condition. '
        'Its full logs are retained here; this directory is excluded from current aggregation. '
        'A completed run with matching inputs is reused without moving it.\n')
    progress('Evaluate','Previous attempt preserved in backup',condition=output.name,archive=target)
    return target

def stop_owned(process):
    # The launcher allows its UE child 20 s to stop. Give that cleanup time
    # to finish before killing the launcher, or its separate UE group survives.
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=45)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def start_world(scene,resource,root,env):
    server_dir = root / 'carla_servers' / worker_name(resource)
    server_dir.mkdir(parents=True, exist_ok=True)
    # Clear readiness from the previous server process.
    (server_dir/'ready.json').unlink(missing_ok=True)
    log=(server_dir/'launcher.log').open('a')
    command=['taskset','-c',resource['cpus'],resource['python'],'-m','benchmark.runtime.server',
        '--scene-dir',scene['scene_dir'],'--output',str(server_dir),
        '--startup-timeout',str(resource.get('startup_timeout_s',3600))]
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();started=time.monotonic();last_message=started
    progress('CARLA','Loading map',scene=scene['scene_dir'],port=resource['port'],server_log=server_dir/'server.log')
    try:
        while not (server_dir/'ready.json').exists():
            if process.poll() is not None:raise RuntimeError('Map launcher exited; inspect '+str(server_dir/'launcher.log'))
            if time.monotonic()-started>resource.get('startup_timeout_s',3600)+660:raise TimeoutError('Map readiness timeout')
            if time.monotonic()-last_message>=60:
                progress('CARLA','Map loading; no experiment has started',wall_s=round(time.monotonic()-started))
                last_message=time.monotonic()
            time.sleep(2)
    except BaseException:stop_owned(process);raise
    progress('CARLA','Map ready',startup_wall_s=round(time.monotonic()-started))
    return process

def score(output,resource,env):
    with (output/'scoring.log').open('a') as log:
        return subprocess.call(['taskset','-c',resource['cpus'],resource.get('metrics_python',resource['python']),
            '-m','benchmark.metrics.evaluation','--run',str(output)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)==0

def method_session(batch_path):
    """Run one method across scenes with the same loaded weights."""
    from benchmark.runtime.run import main as run
    from benchmark.policies.adapters import close_policy
    batch=read(batch_path);policy=None;static_cache={}
    try:
        for index,case in enumerate(batch['cases'],1):
            output=Path(case['output']);output.mkdir(parents=True,exist_ok=True)
            started=time.monotonic()
            cfg=read(case['config'])
            progress('Evaluate','Starting condition',method=cfg['controller'],scene=cfg['scene_id'],deployment=cfg['deployment'],condition=f"{index}/{len(batch['cases'])}",window_s=cfg['duration_s'],requests=len(cfg.get('request_manifest',[])),output=output)
            write(output/'public_status.json',dict(status='running',started_unix=time.time()))
            try:
                policy=run(['--scene-dir',case['scene_dir'],'--config',case['config'],
                    '--output',case['output'],'--port',str(batch['port']),
                    '--carla-root',batch['carla_root'], '--carla-python-api',batch['carla_python_api']],
                    shared_policy=policy,static_cache=static_cache)
                write(output/'public_status.json',dict(status='observed',finished_unix=time.time()))
                progress('Evaluate','Observation saved',scene=cfg['scene_id'],deployment=cfg['deployment'],wall_s=round(time.monotonic()-started),run=output/'run.json')
            except (Exception,SystemExit) as exc:
                write(output/'public_status.json',dict(status='technical_failure',error=str(exc),finished_unix=time.time()))
                import traceback
                traceback.print_exc()
                progress('Evaluate','Technical failure; driving incidents alone do not trigger retries',level='error',scene=cfg['scene_id'],deployment=cfg['deployment'],error=repr(exc),output=output)
                # Let the worker restart the world before retrying this condition.
                return 1
    finally:close_policy(policy)
    return 0

def worker(suite,resource,methods,root,retries):
    resource=expand(resource);city=resource['city'];root=Path(root)
    name=worker_name(resource)
    env=dict(os.environ,**resource.get('env',{}))
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.update(E2F_CPUSET=resource['cpus'],E2F_GPU=str(resource['gpu']),
        E2F_RPC_PORT=str(resource['port']),CARLA_ROOT=resource['carla_root'],UE4_ROOT=resource.get('ue4_root',''),
        CUDA_VISIBLE_DEVICES=str(resource['gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        E2F_SERVER_ARGUMENTS=json.dumps(resource.get('server_arguments',[])))
    env.update(E2F_PROGRESS_LOG=str(root/'logs'/city/'records.log'),E2F_CITY=city,PYTHONUNBUFFERED='1')
    os.environ.update(env)
    progress('Evaluate','Map worker started',pid=os.getpid(),gpu=resource['gpu'],cpus=resource['cpus'],port=resource['port'],methods=','.join(methods),log=env['E2F_PROGRESS_LOG'])
    state=dict(city=city,worker_id=name,pid=os.getpid(),status='starting',methods={},started_unix=time.time())
    state_path=root/'workers'/(name+'.json');server=None;client=None
    write(state_path,state)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for method in methods:
            spec=method_spec(suite,method);all_cases=worker_cases(suite,[method],resource)
            if not all_cases:continue
            if spec.get('rule_controller',{}).get('kind')=='traffic_manager':
                subprocess.run(['taskset','-c',resource['cpus'],resource['python'],'-c','from scipy.spatial import cKDTree; import carla'],env=env,check=True)
            cases=[c for c in all_cases if not c['scene'].get('blocked_reason')]
            blocked=[c['scene']['id'] for c in all_cases if c['scene'].get('blocked_reason')]
            if spec.get('status')!='ready':
                state['methods'][method]=dict(status='blocked',reason=spec.get('reason'),required=len(cases));write(state_path,state);continue
            state['methods'][method]=dict(status='running',required=len(all_cases),completed=0,blocked_scenes=sorted(set(blocked)))
            write(state_path,state)
            progress('Evaluate','Method queue',method=method,conditions=len(all_cases),blocked_scenes=','.join(sorted(set(blocked))) or 'none')
            failures={};exhausted=set();attempt=0
            while True:
                pending=[]
                for case in cases:
                    output=case_paths(root,case)
                    if str(output) in exhausted:continue
                    cfg=configuration(case['scene'],spec,case['deployment'],resource)
                    expected=receipt(case['scene'],cfg)
                    if reusable(output,expected):continue
                    if output.exists() and any(output.iterdir()):
                        archive_condition(output,root)
                    output.mkdir(parents=True,exist_ok=True);write(output/'input_receipt.json',expected)
                    write(output/'submitted_config.json',cfg)
                    pending.append(dict(output=str(output),config=str(output/'submitted_config.json'),scene_dir=case['scene']['scene_dir']))
                if not pending:break
                if server is None:server=start_world(cases[0]['scene'],resource,root,env)
                state.update(status='running',current_method=method,attempt=attempt);write(state_path,state)
                batch_path=root/'workers'/(name+'_'+method+'_batch.json')
                write(batch_path,dict(cases=pending,port=resource['port'],
                                     carla_root=resource['carla_root'],carla_python_api=resource['carla_python_api']))
                python=spec.get('python',resource['python'])
                with (batch_path.with_suffix('.log')).open('a') as log:
                    client=subprocess.Popen(['taskset','-c',resource['cpus'],python,'-m','benchmark',
                        '_session','--batch',str(batch_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while client.poll() is None:
                        try:client.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            running=next((Path(c['output']) for c in pending if (Path(c['output'])/'public_status.json').exists() and read(Path(c['output'])/'public_status.json').get('status')=='running'),None)
                            progress('Evaluate','Worker active',method=method,condition=running.name if running else 'initializing',details=running/'progress.json' if running else batch_path.with_suffix('.log'))
                    code=client.returncode
                if code:
                    stop_owned(server);server=None
                    failed_case=next((c for c in pending if (Path(c['output'])/'public_status.json').exists()
                        and read(Path(c['output'])/'public_status.json').get('status') in ('running','technical_failure')),pending[0])
                    key=failed_case['output'];failures[key]=failures.get(key,0)+1
                    if failures[key]>retries:
                        exhausted.add(key)
                        progress('Evaluate','Technical retry limit reached; preserving this failure and continuing other conditions',level='warning',condition=Path(key).name,attempts=failures[key])
                    attempt+=1
                else:break
            completed=0;scored=0;failed=[]
            for case in cases:
                output=case_paths(root,case)
                if (output/'run.json').exists() and read(output/'run.json').get('execution_completed'):
                    completed+=1
                    ok=score(output,resource,env)
                    if ok:scored+=1
                    progress('Evaluate','Scoring finished',method=method,scene=case['scene']['id'],deployment=case['deployment'],ok=ok,details=output/'scoring.log')
                else:failed.append(case['scene']['id']+'_'+case['deployment'])
            state['methods'][method].update(status='complete' if scored==len(all_cases) else 'incomplete',
                completed=completed,scored=scored,failed=failed,finished_unix=time.time());write(state_path,state)
            progress('Evaluate','Method finished',method=method,completed=completed,scored=scored,required=len(all_cases),failed=','.join(failed) or 'none')
        progress('Evaluate','Map queue finished',methods=','.join(methods))
        state.update(status='finished',finished_unix=time.time());write(state_path,state)
    except BaseException as exc:
        progress('Evaluate','Map worker stopped',level='error',error=repr(exc))
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'technical_failure',error=str(exc));write(state_path,state);raise
    finally:stop_owned(client);stop_owned(server)
    return int(any(m['status']!='complete' for m in state['methods'].values()))
