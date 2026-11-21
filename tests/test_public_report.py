"""Independent small examples for final scoring and incomplete-run handling."""
from pathlib import Path
import pytest
from benchmark.configuration import METRICS
from benchmark.runtime.execution import read
from benchmark.metrics.report import summarize

PARAMETERS = read(Path(__file__).parents[1] / 'configs/score.json')
REGULAR = [key for key in METRICS if key != 'amplification_ratio']


def scene(scene_id, deployment):
    metrics = {key: dict(value=0., status='measured') for key in METRICS}
    metrics['estimated_energy_kwh_per_100_vehicle_km'].update(status='estimated', value=30.)
    metrics['delay_s_per_request'].update(value=60., requests=4)
    metrics['throughput_veh_h']['value'] = 120.  # 2 of 4 requests in 60 s.
    metrics['amplification_ratio']['value'] = 1.
    return dict(scene_id=scene_id, deployment=deployment, window_s=[0., 60.],
                expected_window_s=[0., 60.], observation_complete=True, metrics=metrics,
                coverage=dict(energy_complete=True, energy_vehicle_km=1.))


def data():
    suite = dict(method_order=['Example'], scenes=[dict(id=a,city='PAO',window_s=[0.,60.]) for a in ('S','P')],
                 metric_scenes={key: ['P'] if key=='amplification_ratio' else ['S'] for key in METRICS})
    values = {('Example', a, d): scene(a,d) for a in ('S','P') for d in ('H','M')}
    return suite, values


def test_scores_use_fixed_scene_weights_and_task_specific_denominators():
    suite, values = data()
    # The short platoon never supplies a main-table service throughput.
    values['Example','P','H']['metrics']['throughput_veh_h']['value'] = None
    main = summarize(suite,values,PARAMETERS)['main'][0]
    expected = [0.5,0.5,1.,1.,1.,1.,0.5,0.5]
    assert list(main['normalized_metrics'].values()) == expected
    assert main['score'] == 75.
    assert main['failure_fraction'] == 0.


def test_per_run_scores_at_the_published_reference_values():
    suite, values = data()
    for run in values.values():
        metrics = run['metrics']
        metrics['collision_rate_per_vehicle_km']['value'] = 1.
        metrics['hard_braking_rate_per_vehicle_km']['value'] = 5.
        metrics['low_ttc_exposure_percent']['value'] = 1.
        metrics['speed_sd_mps']['value'] = 2.
    main = summarize(suite, values, PARAMETERS)['main'][0]
    assert list(main['normalized_metrics'].values()) == [.5] * 8
    assert main['score'] == 50.


@pytest.mark.parametrize('distance, expected', [(0., True), (.00099, True), (.001, False)])
def test_minimum_scored_distance_uses_vehicle_km(distance, expected):
    suite, values = data()
    values['Example', 'S', 'H']['coverage']['energy_vehicle_km'] = distance
    cell = summarize(suite, values, PARAMETERS)['conditions'][0]
    assert cell['failed'] is expected


def test_failed_h_keeps_valid_m_but_its_score_weight_stays_one_half():
    suite, values = data()
    values['Example','S','H']['coverage']['energy_vehicle_km'] = .0005
    result = summarize(suite,values,PARAMETERS)
    main = result['main'][0]
    assert main['failure_fraction'] == .25
    assert main['normalized_metrics']['delay_s_per_request'] == .25
    assert main['conditional_raw_metrics']['delay_s_per_request'] == 60.
    assert main['normalized_metrics']['amplification_ratio'] == .5
    assert main['score'] == 40.625


def test_missing_observation_is_pending_not_a_zero_driving_score():
    suite, values = data()
    del values['Example','S','H']
    result = summarize(suite,values,PARAMETERS)
    main = result['main'][0]
    assert main['score'] is None and main['failure_fraction'] is None
    assert main['pending'] == 1 and main['failed'] == 0
    assert main['normalized_metrics']['amplification_ratio'] == .5
    assert main['normalized_metrics']['delay_s_per_request'] is None


def test_observed_departure_is_not_missing_coverage():
    suite, values = data()
    report = values['Example','S','H']
    report['coverage_issues'] = ['invalid_road_intervals']
    for metric in report['metrics'].values():
        metric['calculation_status'] = metric['status']
        metric['status'] = 'observed_with_coverage_limits'
    assert summarize(suite,values,PARAMETERS)['main'][0]['failed'] == 0
    report['coverage_issues'].append('missing_state_intervals')
    assert summarize(suite,values,PARAMETERS)['main'][0]['failed'] == 1


def test_leader_small_excitation_retains_formula_but_flags_scores():
    suite, values = data()
    values['Example','P','H']['metrics']['amplification_ratio'].update(
        value=30., qualifications=['actual_leader_excitation_too_small'])
    result = summarize(suite,values,PARAMETERS)
    assert result['main'][0]['normalized_metrics']['amplification_ratio'] == .25
    assert result['main'][0]['conditional_raw_metrics']['amplification_ratio'] == 1.
    assert result['conditions'][2]['raw']['amplification_ratio'] == 30.


def test_declared_observation_window_cannot_be_replaced_by_a_short_test():
    suite, values = data()
    values['Example','S','H']['expected_window_s'] = [0.,15.]
    with pytest.raises(ValueError,match='window'):
        summarize(suite,values,PARAMETERS)
