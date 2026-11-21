"""Run Ego2Flow on one GPU or evaluate cities in parallel on local GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from benchmark import ROOT
from benchmark.runtime.execution import read, write, method_session, worker, case_paths, cases_for
from benchmark.configuration import defaults, load_suite, prepare, resource_for_city, check_suite


def stop_local(output, city):
    """Signal the recorded map worker, which closes its own client and server."""
    state_path = Path(output).resolve() / 'workers' / (city + '.json')
    if not state_path.exists():
        return
    state = read(state_path)
    if state.get('status') in ('finished', 'interrupted', 'technical_failure'):
        return
    pid = state.get('pid')
    if not pid:
        return
    command_path = Path('/proc') / str(pid) / 'cmdline'
    if not command_path.exists():
        return
    command = command_path.read_bytes().decode().split('\0')
    if ('_map' not in command or '--output' not in command or
            Path(command[command.index('--output') + 1]).resolve() != Path(output).resolve()):
        raise RuntimeError('Recorded PID is not this output directory’s map worker')
    os.kill(pid, signal.SIGTERM)


def launch(resource, args):
    city = resource['city']
    command = ['taskset', '-c', str(resource['cpus']), sys.executable, '-m', 'benchmark', '_map',
               '--suite', str(args.suite.resolve()), '--output', str(args.output.resolve()),
               '--cities', city, '--resource-json', json.dumps(resource),
               '--technical-retries', str(args.technical_retries)]
    if args.methods: command += ['--methods',*args.methods]
    if args.scenes: command += ['--scenes',*args.scenes]
    if args.command == 'doctor': command += ['--check-only']
    log_path = args.output.resolve() / 'logs' / (city+'_launcher.log')
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print(f'{city}: {args.command}; log={log_path}',flush=True)
    with log_path.open('a') as log:
        return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def selected_resources(runtime, mode, cities):
    common = dict(runtime.get('resource', {}))
    maps = common.pop('maps', {})
    if mode == 'sequential':
        assignments = [dict(city=city) for city in cities]
    else:
        assignments = [w for w in runtime['workers'] if w['city'] in cities]
        if sorted(w['city'] for w in assignments) != sorted(cities):
            raise ValueError('Parallel mode needs one worker per selected city')
    resources = []
    for assignment in assignments:
        override = maps.get(assignment['city'], {})
        resource = {**common, **override, **assignment}
        if resource.get('host'):
            raise ValueError('Select a local runtime from configs/runtime.' + mode + '.example.json')
        resource['env'] = {**common.get('env', {}), **override.get('env', {}), **assignment.get('env', {})}
        resources.append(resource)
    if mode == 'parallel':
        if len({int(r['gpu']) for r in resources}) != len(resources):
            raise ValueError('Assign one city per GPU in parallel mode')
        ports = [int(r['port']) + offset for r in resources for offset in (0, 1, 2, 10)]
        if len(ports) != len(set(ports)):
            raise ValueError('CARLA and Traffic Manager port ranges overlap')
    return resources


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True,
                        metavar='{prepare,run,doctor,report,rescore,stop}')
    prep=sub.add_parser('prepare',help='Extract the released six-city scene definitions')
    prep.add_argument('--destination',type=Path)
    descriptions = dict(run='Run configured maps and score completed conditions',
        doctor='Check scene files, environments and dependencies without simulation',
        report='Aggregate saved per-condition evaluations',
        rescore='Recompute metrics from complete raw observations',
        stop='Stop the workers belonging to this output directory')
    for name in ('run','doctor','report','rescore','stop','_map'):
        cmd=sub.add_parser(name, **({'help':descriptions[name]} if name in descriptions else {}))
        cmd.add_argument('--suite',type=Path,default=ROOT/'configs/suite.json')
        cmd.add_argument('--output',type=Path,default=ROOT/'outputs/benchmark')
        cmd.add_argument('--methods',nargs='+')
        cmd.add_argument('--cities',nargs='+')
        cmd.add_argument('--scenes',nargs='+',help='Run selected existing scenes; report coverage for that subset')
        cmd.add_argument('--runtime',type=Path)
        cmd.add_argument('--mode', choices=['sequential', 'parallel'],
                         help='Execution mode (defaults to the runtime configuration)')
        cmd.add_argument('--technical-retries',type=int,default=1)
        if name=='rescore':
            cmd.add_argument('--use-suite-scenes',action='store_true',
                             help='Use the original scenes specified by --suite for runs without saved routes')
        if name=='_map':
            cmd.add_argument('--resource-json',required=True)
            cmd.add_argument('--check-only',action='store_true')
    session=sub.add_parser('_session')
    session.add_argument('--batch',type=Path,required=True)
    args=p.parse_args(argv);defaults()
    if args.command=='prepare': print(prepare(args.destination));return 0
    if args.command=='_session': return method_session(args.batch)
    if args.command=='_map':
        resource=resource_for_city(json.loads(args.resource_json),args.cities[0],args.output)
        suite=load_suite(args.suite,args.cities,args.methods,args.scenes)
        scenes=check_suite(suite,check_models=True)
        if args.check_only:
            from benchmark.runtime.preflight import check_environment
            check_environment(suite,resource,scenes[0]['map'])
            print(f"{resource['city']}: map, launcher and selected method environments checked.");return 0
        try:
            return worker(suite, resource, suite['method_order'], args.output.resolve(), args.technical_retries)
        except KeyboardInterrupt:
            return 130
    suite=load_suite(args.suite,args.cities,args.methods,args.scenes)
    if args.command=='report':
        from benchmark.metrics.report import report
        report(suite,args.output.resolve());return 0
    if args.command=='rescore':
        from benchmark.metrics.evaluation import evaluate
        for city in dict.fromkeys(a['city']for a in suite['scenes']):
            for case in cases_for(suite,suite['method_order'],city):
                output=case_paths(args.output.resolve(),case)
                if (output/'run.json').exists() and read(output/'run.json').get('execution_completed'):
                    scene_dir=Path(case['scene']['scene_dir']) if args.use_suite_scenes else None
                    write(output/'evaluation.json',evaluate(output,scene_dir))
        from benchmark.metrics.report import report
        report(suite,args.output.resolve());return 0
    cities = list(dict.fromkeys(a['city'] for a in suite['scenes']))
    if args.command=='stop':
        for city in cities:
            stop_local(args.output, city)
        return 0
    if not args.runtime:p.error('--runtime is required for '+args.command)
    runtime = read(args.runtime)
    mode = args.mode or ('parallel' if 'workers' in runtime else 'sequential')
    resources = selected_resources(runtime, mode, cities)
    check_suite(suite)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command == 'run':
        write(args.output / 'suite.json', suite)
    city_groups = [resources] if mode == 'parallel' else [[resource] for resource in resources]
    codes = []
    for group in city_groups:
        workers = []
        try:
            for resource in group:
                workers.append(launch(resource, args))
            codes.extend(process.wait() for process in workers)
        except KeyboardInterrupt:
            return 130
        finally:
            running = [process for process in workers if process.poll() is None]
            for process in running:
                process.terminate()
            for process in running:
                process.wait()
        if mode == 'sequential' and codes[-1] in (130, -signal.SIGINT, -signal.SIGTERM):
            return 130
    if args.command=='run':
        from benchmark.metrics.report import report
        report(suite,args.output.resolve())
    return int(any(codes))
