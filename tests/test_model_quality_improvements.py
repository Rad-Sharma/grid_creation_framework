import copy
import numpy as np
import pandas as pd
import pytest
from pypower.case9 import case9
from shapely.geometry import Polygon

from grid_topology import network
from constrained_dispatch import solve_island, assess
from generator_matching import match_station
from operating_scenario import apply_scenario, allocate_sectors
def way(number, nodes, points, **tags):
    return dict(type='way', id=number, nodes=nodes,
                tags={'power':'line', 'voltage':'138000', **tags},
                geometry=[dict(lon=x, lat=y) for x,y in points])


def section_fixture():
    boundary = Polygon([(-101,30),(-99,30),(-99,32),(-101,32)])
    station = dict(type='way',id=987654,tags={'power':'substation','voltage':'138000;69000'},
        geometry=[dict(lon=x,lat=y) for x,y in [(-100.01,30.99),(-99.99,30.99),
                  (-99.99,31.01),(-100.01,31.01),(-100.01,30.99)]])
    a=way(1,[1,2],[(-100.5,31),(-100.005,31)],transmission_system='ERCOT')
    b=way(2,[3,4],[(-99.995,31),(-99.5,31)],transmission_system='SPP')
    tie=way(3,[2,3],[(-100.005,31),(-99.995,31)],note='Synchronous tie. Normally open.')
    return boundary,station,a,b,tie


def test_station_merging_cannot_bypass_open_tie():
    boundary,station,a,b,tie = section_fixture()
    buses,branches,_ = network([station,a,b,tie],boundary)
    assert branches[0]['tbus'] != branches[1]['fbus']
    assert buses[buses.kind.eq('substation')].section.nunique()==2


def test_shared_engine_rejects_degree_based_length_calculation():
    from grid_topology import network as shared_network
    boundary,station,a,b,_=section_fixture()
    with pytest.raises(ValueError,match='meter units'):
        shared_network([station,a,b],boundary,metric_crs=4326)


def test_export_preserves_case_base_mva(tmp_path):
    from case_utils import write_case
    from ensure_feasibility import load_matpower
    case=case9();case['baseMVA']=50.
    path=tmp_path/'base50.m'
    write_case(case,path,'unit test')
    assert load_matpower(path)['baseMVA']==50.


def test_station_separates_conflicting_systems_even_without_open_tag():
    boundary,station,a,b,_ = section_fixture()
    _,branches,_ = network([station,a,b],boundary)
    assert branches[0]['tbus'] != branches[1]['fbus']


def test_transformer_requires_evidence_and_preserves_supplied_rating():
    boundary,station,a,b,_ = section_fixture()
    b['tags']['voltage']='69000'
    b['tags']['transmission_system']='ERCOT'
    _,branches,omitted = network([station,a,b],boundary)
    assert len(branches)==2
    assert omitted.reason.str.contains('unverified transformer').any()
    evidence={'connections':[dict(kind='transformer',source='test equipment sheet',id='xf1',
        **{'from':dict(station='way/987654',kv=138,section='default'),
           'to':dict(station='way/987654',kv=69,section='default')},
        r_pu=.001,x_pu=.02,rate_mva=85)]}
    _,branches,_ = network([station,a,b],boundary,evidence=evidence)
    assert branches[-1]['kind']=='reviewed_transformer'
    assert branches[-1]['rateA']==85
    assert branches[-1]['x']==.02
    bad=copy.deepcopy(evidence);bad['connections'][0]['source']=''
    with pytest.raises(ValueError,match='source'):
        network([station,a,b],boundary,evidence=bad)


def test_generator_matching_rejects_distant_voltage_trap_and_ambiguous_station():
    stations=pd.DataFrame(dict(bus_i=[1,2],x=[1000,12000],y=[0,0],base_kV=[138,345],
        longitude=[-100,-99],latitude=[31,31],interconnection=['Texas','Texas']))
    selected,reason,_=match_station(stations,np.array([0,0]),345,'Texas')
    assert selected is None
    stations.loc[1,['x','base_kV']]=[1100,138]
    selected,reason,alternatives=match_station(stations,np.array([0,0]),138,'Texas')
    assert selected is None and 'ambiguous' in reason and len(alternatives)==2
    stations.loc[1,'interconnection']='Eastern'
    selected,_,_=match_station(stations,np.array([0,0]),138,'Texas')
    assert selected.bus_i==1


def scenario_spec():
    return dict(source='synthetic unit test, not measured data',timestamp='2024-07-01T12:00:00-05:00',
        duration_hours=1,replace_all_loads=True,
        sector_loads=[dict(bus=5,sector='industrial',mw=20,power_factor=.95)],
        generator_availability=[dict(row_1based=i,fraction=.5) for i in range(1,4)],
        storage=[dict(bus=5,mw=10,initial_mwh=30,capacity_mwh=50,charge_limit_mw=20,
            discharge_limit_mw=20,charge_efficiency=.9,discharge_efficiency=.9,
            qmin_mvar=-5,qmax_mvar=5,source='test storage')])


def test_operating_scenario_tracks_storage_and_retains_input_case():
    case=case9();original=case['bus'].copy();spec=scenario_spec()
    result,report=apply_scenario(case,spec)
    assert result['bus'][:,2].sum()==20
    assert np.isclose(report['storage'][0]['final_mwh'],30-10/.9)
    np.testing.assert_array_equal(case['bus'],original)
    np.testing.assert_allclose(result['gen'][:3,8],case['gen'][:,8]/2)
    assert len(result['gen'])==len(result['gencost'])==4
    spec['storage'][0]['initial_mwh']=1
    with pytest.raises(ValueError,match='energy bounds'):
        apply_scenario(case,spec)


def test_scenario_rejects_missing_availability():
    spec=scenario_spec();spec['generator_availability'].pop()
    with pytest.raises(ValueError,match='every active'):
        apply_scenario(case9(),spec)


def test_sector_weights_preserve_industrial_concentration():
    import geopandas as gpd
    buses=pd.DataFrame(dict(bus_i=[1,2],longitude=[-100,-99.9],latitude=[31,31]))
    counties=gpd.GeoDataFrame(dict(GEOID=['48001']),geometry=[Polygon([(-101,30),(-99,30),(-99,32),(-101,32)])],crs=4326)
    totals=pd.DataFrame(dict(GEOID=['48001','48001'],sector=['residential','industrial'],mw=[20,80],source=['test']*2))
    weights=pd.DataFrame(dict(bus_i=[1,2,2],sector=['residential','residential','industrial'],weight=[1,1,1],source=['test']*3))
    load,_=allocate_sectors(buses,counties,totals,weights)
    np.testing.assert_allclose(load,[10,90])
    with pytest.raises(ValueError,match='weights'):
        allocate_sectors(buses,counties,totals,weights.iloc[:2])


def test_constrained_opf_passes_without_relaxing_equipment():
    case=case9();original=copy.deepcopy(case)
    solved,report=solve_island(case)
    assert report['passes'],report
    assert solved is not None
    for key in ('bus','gen','branch','gencost'):
        np.testing.assert_array_equal(case[key],original[key])
    solved['bus'][0,12]=solved['bus'][0,7]+.01
    assert not assess(solved)['passes']


def test_validator_zero_rating_means_unlimited():
    solved,report=solve_island(case9())
    assert report['passes']
    solved['branch'][:,5]=0
    assert assess(solved)['thermal_mva']==0
