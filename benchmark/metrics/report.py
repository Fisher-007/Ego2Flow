"""Per-run metrics, fixed-weight normalized scores, and explicit coverage."""
import csv
import json
import math
from pathlib import Path
from benchmark.configuration import METRICS
from benchmark import ROOT
from benchmark.runtime.execution import read, write, case_paths


def normalized(metric, value, parameters, requests=None, duration=None):
    if not math.isfinite(value): raise ValueError('Non-finite '+metric)
    if metric=='throughput_veh_h':
        if not requests or not duration or duration<=0:raise ValueError('Throughput requires scheduled demand and duration')
        return min(1., max(0., value/(3600.*requests/duration)))
    if metric=='estimated_energy_kwh_per_100_vehicle_km': value=max(0., value)
    if value<0:raise ValueError('Negative '+metric)
    return 1./(1.+value/parameters['scales'][metric])


def eligible_value(evaluation, key):
    metric = evaluation['metrics'][key]
    issues = set(evaluation.get('coverage_issues', []))
    issues.discard('invalid_road_intervals')
    if key == 'throughput_veh_h': issues.discard('missing_state_intervals')
    if key != 'speed_sd_mps': issues.discard('ungrouped_vehicle_exposure')
    status = metric.get('calculation_status', metric.get('status'))
    if key == 'estimated_energy_kwh_per_100_vehicle_km':
        usable = status in ('estimated', 'estimated_contact_qualified') and evaluation.get('coverage', {}).get('energy_complete', False)
    else:
        usable = status == 'measured'
    return metric.get('value') if not issues and usable else None


def score_condition(evaluation, applicable, parameters):
    """Score observed failures as zero and leave incomplete runs pending."""
    if evaluation is None or not evaluation.get('observation_complete'):
        return dict(status='pending',failed=None,reasons=['missing_or_incomplete_observation'],
                    raw={k:None for k in METRICS},normalized={k:None for k in METRICS})
    metrics=evaluation['metrics'];reasons=[]
    raw={k:eligible_value(evaluation,k) for k in METRICS}
    for k in applicable:
        if raw[k] is None or not math.isfinite(raw[k]): reasons.append('undefined:'+k)
    if 'amplification_ratio' in applicable:
        peak=metrics['amplification_ratio']
        loss=peak.get('actual_leader_speed_loss_mps')
        threshold=parameters['minimum_leader_speed_loss_mps']
        small=(loss<threshold if loss is not None else
               'actual_leader_excitation_too_small' in peak.get('qualifications',[]))
        if small:reasons.append(f'leader_speed_loss_below_{threshold:g}_mps')
    else:
        distance=evaluation.get('coverage',{}).get('energy_vehicle_km')
        if distance is not None and distance<parameters['minimum_evaluated_distance_km']:
            reasons.append('scored_distance_below_1_vehicle_m')
    failed=bool(reasons); duration=evaluation['window_s'][1]-evaluation['window_s'][0]
    n=metrics['delay_s_per_request'].get('requests')
    scores={k:None for k in METRICS}
    for k in applicable:scores[k]=0. if failed else normalized(k,raw[k],parameters,n,duration)
    return dict(status='observed',failed=failed,reasons=reasons,raw=raw,normalized=scores)


def mean_complete(values):
    return math.fsum(values)/len(values) if values and all(v is not None for v in values) else None


def summarize(suite, evaluations, parameters):
    """Aggregate H and M results by scene and method."""
    conditions=[];scenes=[];main=[]
    for method in suite['method_order']:
        for scene in suite['scenes']:
            applicable=[k for k in METRICS if scene['id'] in suite['metric_scenes'][k]]
            cells={}
            for deployment in ('H','M'):
                evaluation=evaluations.get((method,scene['id'],deployment))
                if evaluation is not None:
                    if (evaluation['scene_id'],evaluation['deployment'])!=(scene['id'],deployment):raise ValueError('Evaluation identity mismatch')
                    if evaluation['expected_window_s']!=scene['window_s']:raise ValueError('Evaluation window differs from suite')
                cell=score_condition(evaluation,applicable,parameters);cells[deployment]=cell
                conditions.append(dict(method=method,scene_id=scene['id'],city=scene['city'],deployment=deployment,**cell))
            # Flagged runs count as zero in the normalized H/M average and drop out of conditional raw means.
            score={k:mean_complete([cells[d]['normalized'][k]for d in ('H','M')])for k in METRICS}
            conditional={}
            raw_balanced={}
            for k in METRICS:
                retained=[cells[d]['raw'][k]for d in ('H','M')if cells[d]['status']=='observed' and not cells[d]['failed'] and k in applicable]
                conditional[k]=mean_complete(retained)
                raw_balanced[k]=mean_complete([cells[d]['raw'][k] if cells[d]['status']=='observed' and k in applicable else None for d in ('H','M')])
            scenes.append(dict(method=method,scene_id=scene['id'],H=cells['H'],M=cells['M'],normalized=score,raw_balanced=raw_balanced,conditional_raw=conditional))
        selected=[r for r in scenes if r['method']==method]
        scores={k:mean_complete([r['normalized'][k]for r in selected if r['scene_id']in suite['metric_scenes'][k]])for k in METRICS}
        raw={k:mean_complete([r['raw_balanced'][k]for r in selected if r['scene_id']in suite['metric_scenes'][k]])for k in METRICS}
        conditional={k:mean_complete([r['conditional_raw'][k]for r in selected if r['scene_id']in suite['metric_scenes'][k]and r['conditional_raw'][k]is not None])for k in METRICS}
        runs=[r for r in conditions if r['method']==method]; pending=sum(r['status']=='pending'for r in runs);failed=sum(r['failed']is True for r in runs)
        main.append(dict(method=method,scheduled=len(runs),observed=len(runs)-pending,pending=pending,failed=failed,
                         failure_fraction=failed/len(runs)if not pending else None,normalized_metrics=scores,
                         score=parameters['total_scale']*math.fsum(scores.values())if all(v is not None for v in scores.values())else None,
                         raw_metrics=raw,conditional_raw_metrics=conditional))
    return dict(parameters=parameters,metric_scenes=suite['metric_scenes'],main=main,scenes=scenes,conditions=conditions)


def report(suite, root):
    root=Path(root);evaluations={}
    for method in suite['method_order']:
        for scene in suite['scenes']:
            for deployment in ('H','M'):
                folder=case_paths(root,dict(method=method,scene=scene,deployment=deployment))
                if not (folder/'evaluation.json').exists():continue
                evaluation=read(folder/'evaluation.json')
                # Complete simulator execution is recorded separately from driving success.
                if not (folder/'run.json').exists() or not read(folder/'run.json').get('execution_completed'):continue
                evaluations[method,scene['id'],deployment]=evaluation
    result=summarize(suite,evaluations,read(ROOT/'configs/score.json'))
    target=root/'report';target.mkdir(parents=True,exist_ok=True);write(target/'results.json',result)
    names={m:read(suite['methods'][m])['name']for m in suite['method_order']}
    for filename,field in [('scores.csv','normalized_metrics'),('raw_metrics.csv','raw_metrics'),('conditional_raw.csv','conditional_raw_metrics')]:
        with (target/filename).open('w',newline='')as f:
            writer=csv.writer(f);writer.writerow(['Method',*METRICS,'Failure fraction','Total score','Observed','Scheduled'])
            for r in result['main']:writer.writerow([names[r['method']],*[r[field][k]for k in METRICS],r['failure_fraction'],r['score'],r['observed'],r['scheduled']])
    with (target/'conditions.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['Method','Scene','Deployment','Status','Flagged','Reasons',*METRICS,*['score:'+k for k in METRICS]])
        for r in result['conditions']:w.writerow([names[r['method']],r['scene_id'],r['deployment'],r['status'],r['failed'],';'.join(r['reasons']),*[r['raw'][k]for k in METRICS],*[r['normalized'][k]for k in METRICS]])
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    labels=['delay','throughput','collision','low TTC','hard braking','speed SD','peak loss','energy']
    lines=['# Benchmark results','','The eight metric scores range from 0 to 1 and the total from 0 to 100. Higher scores are better. H and M each contribute half of each scene’s score. Applicable scenes have equal weight.','','| Method | '+' | '.join(labels)+' | Failure fraction | Total score | Observed / scheduled |','|---|'+'---:|'*11]
    for r in result['main']:lines.append('| '+names[r['method']]+' | '+' | '.join(fmt(r['normalized_metrics'][k])for k in METRICS)+' | '+fmt(r['failure_fraction'])+' | '+fmt(r['score'])+f" | {r['observed']}/{r['scheduled']} |")
    lines+=['','See `conditions.csv` for run status and failure reasons. `raw_metrics.csv` gives metrics in their original units, including collisions and hard-braking events per vehicle-km. `conditional_raw.csv` includes only unflagged deployments.','']
    (target/'RESULTS.md').write_text('\n'.join(lines));print(target/'RESULTS.md');return result
