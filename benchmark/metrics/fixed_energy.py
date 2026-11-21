"""Estimate energy demand from recorded trajectories."""
import json
import math

import numpy as np

from benchmark import ROOT


# Wheel work follows SUMO v1_23_1 HelpersEnergy (EPL-2.0/GPL-2.0-or-later).
# https://github.com/eclipse-sumo/sumo/blob/v1_23_1/src/utils/emissions/HelpersEnergy.cpp
def estimate_fixed_cycle(times, speeds, grades, settings, initial_soc=None, collision_intervals=None, anomalies=None, positions=None, scoring_intervals=None):
    """Integrate battery energy over the recorded trajectory intervals."""
    t,v,g=(np.asarray(x,dtype=float) for x in (times,speeds,grades))
    if len(t)<2:return dict(status='insufficient_observation',energy_kwh=None)
    if not (len(t)==len(v)==len(g)) or not all(np.isfinite(x).all() for x in (t,v,g)) or np.any(np.diff(t)<=0) or np.any(v<0):
        return dict(status='invalid_energy_input',energy_kwh=None)
    if settings.get('reference_parameters'):
        p=json.loads((ROOT / settings['reference_parameters']).read_text())['parameters']
        if (p['resource'],p['parameter_version'])!=(settings['vehicle_resource'],settings['version']):
            raise ValueError('Explicit energy parameters and declared reference differ')
    else:
        raise ValueError('Set reference_parameters to the released reference vehicle JSON')
    collision_intervals=collision_intervals or [False]*(len(t)-1)
    anomalies=anomalies or [False]*(len(t)-1)
    scoring_intervals=[True]*(len(t)-1) if scoring_intervals is None else list(scoring_intervals)
    soc=settings['initial_soc'] if initial_soc is None else initial_soc
    initial=soc;net=aux=regen=traction=friction=contact_kinetic=0.
    violations=[];rows=[];power_episodes=[]
    for i,dt in enumerate(np.diff(t)):
        scored=scoring_intervals[i]
        v0,v1=v[i:i+2];mean=(v0+v1)/2;grade=(g[i]+g[i+1])/2;theta=math.atan(grade)
        acceleration=(v1-v0)/dt
        kinetic=.5*(p['mass_kg']+p['rotating_mass_kg'])*(v1*v1-v0*v0)
        # Integrate v^3 exactly for the declared linear interval interpolation.
        drag=.5*p['air_density_kg_m3']*p['drag_coefficient']*p['frontal_area_m2']*dt*(v0**3+v0*v0*v1+v0*v1*v1+v1**3)/4
        rolling=p['mass_kg']*p['gravity_mps2']*p['rolling_coefficient']*math.cos(theta)*mean*dt
        ascent=p['mass_kg']*p['gravity_mps2']*math.sin(theta)*mean*dt
        collision=bool(collision_intervals[i])
        # Separate impact kinetic work from propulsion; retain road load and auxiliaries.
        contact_kinetic+=abs(kinetic) if collision and scored else 0.
        scenario=settings.get('contact_sensitivity_case')
        contact_kinetic_work=(max(0.,kinetic) if scenario=='positive_work_only' else
                              min(0.,kinetic) if scenario=='negative_work_only' else 0.)
        work=(contact_kinetic_work if collision else kinetic)+drag+rolling+ascent
        wheel_power=work/dt;eta_t=p['transmission_efficiency'];aux_j=p['auxiliary_w']*dt
        if wheel_power>=0:
            mechanical=wheel_power/eta_t
            # The efficiency lookup stops at rated power. Mechanical demand can exceed it.
            fraction=min(1.,mechanical/p['motor_power_w'])
            eta=(float(np.interp(fraction,p['efficiency_fraction'],p['efficiency'])) if p['efficiency_interpolation']=='Linear'
                else p['efficiency'][max(0,int(np.searchsorted(p['efficiency_fraction'],fraction,side='right'))-1)])
            electrical=mechanical/eta
            regen_j=0.;friction_j=0.
            if scored:traction+=electrical*dt
        else:
            mechanical=min(-wheel_power*eta_t,p['motor_power_w'])
            eta=p['efficiency'][0]
            electrical_recovery=min(mechanical*eta,p['battery_power_w'],
                max(0.,(p['maximum_soc']-soc)*p['battery_capacity_j'])/(p['battery_efficiency']*dt))
            if collision and scenario!='negative_work_only':electrical_recovery=0.
            electrical=-electrical_recovery
            regen_j=electrical_recovery*dt
            friction_j=max(0.,-work-regen_j/(eta*eta_t))
        terminal_j=electrical*dt+aux_j
        chemical_j=terminal_j/p['battery_efficiency'] if terminal_j>=0 else terminal_j*p['battery_efficiency']
        soc-=chemical_j/p['battery_capacity_j']
        reasons=[];warnings=[]
        if acceleration>p['maximum_traction_acceleration_mps2']+1e-6:warnings.append('reference_tire_traction_exceeded')
        if not collision and -acceleration>p['maximum_friction_deceleration_mps2']+1e-6:warnings.append('reference_tire_braking_exceeded')
        motor_excess=max(0.,wheel_power/eta_t-p['motor_power_w'])
        battery_excess=max(0.,terminal_j/dt-p['battery_power_w'])
        if motor_excess>1e-6:warnings.append('motor_efficiency_endpoint_extension')
        if battery_excess>1e-6:warnings.append('reference_battery_power_exceeded')
        if soc<p['minimum_soc']-1e-8:reasons.append('reference_battery_depleted')
        if anomalies[i]:reasons.append('position_velocity_inconsistent')
        if collision and abs(kinetic)>1e-6:reasons.append('contact_energy_attribution_uncertain')
        row=dict(start_s=float(t[i]),end_s=float(t[i+1]),speed_start_mps=float(v0),speed_end_mps=float(v1),
            acceleration_mps2=float(acceleration),grade=float(grade),wheel_power_w=float(wheel_power),
            battery_terminal_power_w=float(terminal_j/dt),collision_no_regeneration=collision,reasons=reasons,warnings=warnings,
            motor_excess_w=float(motor_excess),battery_excess_w=float(battery_excess),
            propulsion_terminal_energy_kwh=max(0.,electrical)*dt/3.6e6,
            regeneration_terminal_energy_kwh=regen_j/3.6e6,
            auxiliary_terminal_energy_kwh=aux_j/3.6e6,
            net_terminal_energy_kwh=terminal_j/3.6e6,
            kinetic_change_kwh=kinetic/3.6e6,distance_m=mean*dt)
        if positions is not None:row.update(position_start_m=positions[i],position_end_m=positions[i+1])
        if not scored:continue  # Unscored intervals update SOC only.
        rows.append(row)
        if reasons:violations.append(row)
        if motor_excess>1e-6 or battery_excess>1e-6:
            # Merge contiguous exceedances, retaining times, positions and energy.
            if not power_episodes or abs(power_episodes[-1]['end_s']-t[i])>1e-8:
                power_episodes.append(dict(start_s=float(t[i]),end_s=float(t[i]),duration_s=0.,
                    start_position_m=positions[i] if positions is not None else None,
                    maximum_wheel_power_w=0.,maximum_battery_terminal_power_w=0.,
                    propulsion_terminal_energy_kwh=0.,battery_excess_kwh=0.,motor_excess_kwh=0.))
            event=power_episodes[-1];event['end_s']=float(t[i+1]);event['duration_s']+=float(dt)
            event['end_position_m']=positions[i+1] if positions is not None else None
            event['maximum_wheel_power_w']=max(event['maximum_wheel_power_w'],float(wheel_power))
            event['maximum_battery_terminal_power_w']=max(event['maximum_battery_terminal_power_w'],float(terminal_j/dt))
            event['propulsion_terminal_energy_kwh']+=row['propulsion_terminal_energy_kwh']
            event['battery_excess_kwh']+=float(battery_excess*dt/3.6e6)
            event['motor_excess_kwh']+=float(motor_excess*dt/3.6e6)
        net+=terminal_j;aux+=aux_j;regen+=regen_j;friction+=friction_j
    invalid=sorted({reason for r in violations for reason in r['reasons']})
    status=('estimated' if not invalid else 'energy_input_motion_anomaly' if 'position_velocity_inconsistent' in invalid
        else 'energy_contact_attribution_uncertain' if 'contact_energy_attribution_uncertain' in invalid
        else 'energy_battery_depleted' if 'reference_battery_depleted' in invalid
        else 'energy_reference_power_range_exceeded')
    result=dict(status=status,energy_kwh=net/3.6e6 if not invalid else None,
        **({'interval_details':rows} if settings.get('include_interval_details') else {}),
        diagnostic_demand_kwh=net/3.6e6,
        diagnostic_only_reason=None if not invalid else ', '.join(invalid),invalid_reasons=invalid,
        auxiliary_terminal_energy_kwh=aux/3.6e6,regenerated_terminal_energy_kwh=regen/3.6e6,
        traction_terminal_energy_kwh=traction/3.6e6,friction_braking_energy_kwh=friction/3.6e6,
        unattributed_contact_kinetic_kwh=contact_kinetic/3.6e6,
        initial_soc=initial,final_soc=soc,elapsed_s=float(t[-1]-t[0]),scored_elapsed_s=sum(r['end_s']-r['start_s'] for r in rows),
        observed_distance_m=sum(r['distance_m'] for r in rows),
        maximum_speed_error_mps=0.,maximum_time_error_s=0.,
        collision_no_regeneration_s=sum(r['end_s']-r['start_s'] for r in rows if r['collision_no_regeneration']),
        reference_driving_warning_seconds=sum(r['end_s']-r['start_s'] for r in rows if any(w.startswith('reference_tire_') for w in r['warnings'])),
        qualifications=sorted({w for r in rows for w in r['warnings'] if w in ('reference_battery_power_exceeded','motor_efficiency_endpoint_extension')}),
        violation_count=len(violations),violation_seconds=sum(r['end_s']-r['start_s'] for r in violations),
        first_violations=violations[:12],power_exceedance_episodes=power_episodes,
        motor_map_extension_vehicle_s=sum(r['end_s']-r['start_s'] for r in rows if r['motor_excess_w']>1e-6),
        motor_map_peak_excess_fraction=max(r['motor_excess_w'] for r in rows)/p['motor_power_w'],
        motor_map_extension_drive_kwh=sum(r['propulsion_terminal_energy_kwh'] for r in rows if r['motor_excess_w']>1e-6),
        maximum_wheel_power_w=max(r['wheel_power_w'] for r in rows),
        maximum_battery_terminal_power_w=max(r['battery_terminal_power_w'] for r in rows),
        maximum_acceleration_mps2=max(r['acceleration_mps2'] for r in rows),minimum_acceleration_mps2=min(r['acceleration_mps2'] for r in rows),
        model=p,method='fixed_trajectory_interval_work',eligibility='fixed_trajectory_endpoint_hold',
        sampling='Recorded interval endpoints with piecewise-linear speed.',
        definition='Estimated battery-terminal demand over scored vehicles and intervals, with limited regeneration. Unscored intervals update SOC. Above reference power, retain full demand at endpoint motor efficiency and record exceedance exposure.')
    if settings.get('quantify_contact_sensitivity'):
        alternatives={}
        for case in ('positive_work_only','negative_work_only'):
            if any(collision_intervals):
                alternative=estimate_fixed_cycle(times,speeds,grades,
                    {**settings,'quantify_contact_sensitivity':False,'include_interval_details':False,'contact_sensitivity_case':case},
                    initial_soc=initial_soc,collision_intervals=collision_intervals,anomalies=anomalies,scoring_intervals=scoring_intervals)
                alternatives[case]=alternative['diagnostic_demand_kwh']
            else:alternatives[case]=result['diagnostic_demand_kwh']
        result['contact_sensitivity_net_kwh']=alternatives
        result['contact_sensitivity_meaning']='Sensitivity to positive-only or negative-only impact kinetic-work attribution, relative to the estimate that excludes impact kinetic work.'
    return result


def summarize_scene_energy(segments, distance_km, state_data_complete, contact_limit_fraction=None, collision_stream_complete=True):
    """Summarize scene energy and consumption per 100 vehicle-km."""
    complete=state_data_complete and bool(segments)
    fields={
        'drive_kwh':'traction_terminal_energy_kwh',
        'regeneration_kwh':'regenerated_terminal_energy_kwh',
        'auxiliary_kwh':'auxiliary_terminal_energy_kwh',
        'diagnostic_net_kwh':'diagnostic_demand_kwh',
        'friction_braking_kwh':'friction_braking_energy_kwh',
        'unattributed_contact_kinetic_kwh':'unattributed_contact_kinetic_kwh'}
    result={name:(sum(s[key] for s in segments) if complete and all(key in s for s in segments) else None)
            for name,key in fields.items()}
    valid=complete and collision_stream_complete and all(s['status']=='estimated' for s in segments)
    episodes=[dict(request_id=s['request_id'],**event) for s in segments for event in s.get('power_exceedance_episodes',[])]
    result.update(vehicle_count=len({s['request_id'] for s in segments}),
        vehicle_s=sum(s.get('scored_elapsed_s',s['end_s']-s['start_s']) for s in segments),distance_vehicle_km=distance_km,
        state_data_complete=complete,model_applicable=valid,
        motor_map_extension_vehicle_s=sum(s.get('motor_map_extension_vehicle_s',0.) for s in segments),
        motor_map_peak_excess_fraction=max((s.get('motor_map_peak_excess_fraction',0.) for s in segments),default=0.),
        motor_map_extension_drive_kwh=sum(s.get('motor_map_extension_drive_kwh',0.) for s in segments),
        net_kwh=result['diagnostic_net_kwh'] if valid else None,
        kwh_per_100_vehicle_km=100*result['diagnostic_net_kwh']/distance_km if valid and distance_km>0 else None,
        diagnostic_kwh_per_100_vehicle_km=100*result['diagnostic_net_kwh']/distance_km if result['diagnostic_net_kwh'] is not None and distance_km>0 else None,
        invalid_request_ids=sorted({s['request_id'] for s in segments if s['status']!='estimated'}),
        invalid_reasons=sorted({reason for s in segments for reason in s.get('invalid_reasons',[s['status']] if s['status']!='estimated' else [])}),
        qualifications=sorted({q for s in segments for q in s.get('qualifications',[])}),
        power_exceedance_request_ids=sorted({e['request_id'] for e in episodes}),
        power_exceedance_vehicle_s=sum(e['duration_s'] for e in episodes),
        energy_during_power_exceedance_kwh=sum(e['propulsion_terminal_energy_kwh'] for e in episodes),
        battery_excess_kwh=sum(e['battery_excess_kwh'] for e in episodes),
        motor_excess_kwh=sum(e['motor_excess_kwh'] for e in episodes),power_exceedance_episodes=episodes)
    if complete and all('contact_sensitivity_net_kwh' in s for s in segments):
        base=result['diagnostic_net_kwh']
        alternatives={case:sum(s['contact_sensitivity_net_kwh'][case] for s in segments)
                      for case in ('positive_work_only','negative_work_only')}
        fraction=max(abs(x-base)/abs(base) for x in alternatives.values()) if base and base>0 else None
        result['contact_sensitivity']=dict(net_kwh=alternatives,maximum_relative_change=fraction)
        if contact_limit_fraction is not None:
            result['contact_sensitivity']['reporting_limit_fraction']=contact_limit_fraction
        qualified=(collision_stream_complete and
            result['invalid_reasons']==['contact_energy_attribution_uncertain'])
        if qualified:
            result.update(model_applicable=True,net_kwh=base,
                kwh_per_100_vehicle_km=100*base/distance_km if distance_km>0 else None,
                contact_qualified_estimate=True)
            result['qualifications'].append('contact_kinetic_attribution_sensitivity_disclosed')
    if not collision_stream_complete:
        result['invalid_reasons'].append('collision_event_stream_incomplete')
    return result
