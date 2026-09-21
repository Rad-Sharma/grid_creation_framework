import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from grid_topology import network
from substation_audit import (collect_substations, duplicate_candidates, missing_candidates,
                              reference_gaps, run_audit)

BOUNDARY = Polygon([(-101,30),(-99,30),(-99,32),(-101,32)])


def station(number=1, kind='way', name='Sample Station', voltage='138000', operator='Utility A'):
    tags=dict(power='substation',name=name,voltage=voltage,operator=operator)
    if kind=='node':
        return dict(type=kind,id=number,tags=tags,lon=-100,lat=31)
    return dict(type=kind,id=number,tags=tags,
        geometry=[dict(lon=x,lat=y) for x,y in [(-100.001,30.999),(-99.999,30.999),
                    (-99.999,31.001),(-100.001,31.001),(-100.001,30.999)]])


def test_point_polygon_duplicate_is_flagged_not_merged():
    elements=[station(),station(2,'node',name='SAMPLE-STATION')]
    records=collect_substations(elements,BOUNDARY)
    candidates=duplicate_candidates(records)
    assert len(records)==2 and len(candidates)==1
    assert candidates[0]['category']=='likely_duplicate'
    assert candidates[0]['point_in_polygon']
    assert candidates[0]['decision']=='unreviewed'
    assert candidates[0]['topology_action'].startswith('none')


@pytest.mark.parametrize('changes,category',[
    ({'voltage':'345000'},'possible_distinct_voltage_yards'),
    ({'operator':'Utility B'},'conflicting_operators'),
    ({'name':'Different Yard'},'possible_duplicate_or_adjacent_sections'),
])
def test_overlap_does_not_erase_distinct_yards(changes,category):
    candidates=duplicate_candidates(collect_substations([station(),station(2,**changes)],BOUNDARY))
    assert candidates[0]['category']==category


def test_valid_relation_member_alias_and_invalid_parent_fallback():
    child=station()
    parent=dict(type='relation',id=3,tags=child['tags'],members=[
        dict(type='way',ref=1,role='outer',geometry=child['geometry'])])
    records=collect_substations([parent,child],BOUNDARY)
    assert next(r for r in records if r['osm_id']=='way/1')['parent_relations']==['relation/3']
    assert duplicate_candidates(records)[0]['category']=='relation_member_alias'
    parent['members'][0].pop('geometry')
    records=collect_substations([parent,child],BOUNDARY)
    assert next(r for r in records if r['osm_id']=='way/1')['parent_relations']==[]
    line=dict(type='way',id=4,nodes=[10,11],tags={'power':'line','voltage':'138000'},
              geometry=[dict(lon=-100,lat=31),dict(lon=-100.1,lat=31)])
    buses,_,_=network([parent,child,line],BOUNDARY)
    assert buses.osm_id.eq('way/1').any()


def test_review_decisions_persist_but_do_not_merge_electrical_records():
    records=collect_substations([station(),station(2,'node')],BOUNDARY)
    decision=dict(left='way/1',right='node/2',decision='separate_sites',source='review drawing 2026-01')
    result=duplicate_candidates(records,decisions=[decision])
    assert result[0]['decision']=='separate_sites'
    assert len(records)==2
    decision['source']=''
    with pytest.raises(ValueError,match='source'):
        duplicate_candidates(records,decisions=[decision])


def gap_network():
    buses=pd.DataFrame(dict(bus_i=[1,2,3,4,5,6],
        longitude=[-100,-99.999,-100.1,-99.9,-100.05,-99.000001],latitude=[31]*6,
        base_kV=[138]*6,kind=['osm_junction','osm_junction','substation','substation','osm_junction','osm_junction']))
    branches=pd.DataFrame(dict(fbus=[1,2,3,5,6],tbus=[3,4,5,4,4]))
    return buses,branches


def test_gap_clustering_excludes_degree_two_and_border_endpoints():
    buses,branches=gap_network()
    candidates=missing_candidates([],buses,branches,BOUNDARY)
    assert len(candidates)==1
    assert candidates[0]['bus_ids']==[1,2]
    assert candidates[0]['category']=='possible_missing_substation_or_line_connection'
    assert candidates[0]['priority']=='review'
    records=collect_substations([station(voltage='11000')],BOUNDARY)
    candidates=missing_candidates(records,buses,branches,BOUNDARY)
    assert candidates[0]['category']=='existing_station_unmatched'
    assert candidates[0]['nearby_source_substations']==['way/1']


def test_reference_comparison_distinguishes_scope_and_absence():
    reference=gpd.GeoDataFrame(dict(reference_id=['a','b','c'],source=['independent example']*3),
        geometry=[Point(-100,31),Point(-99.5,31),Point(-98,31)],crs=4326)
    records=collect_substations([station()],BOUNDARY)
    result=reference_gaps(records,reference,boundary=BOUNDARY)
    assert [r['status'] for r in result]==['spatial_candidate_match','not_found_within_search_radius','outside_study_boundary']


def test_outputs_repeat_when_source_and_bus_order_change(tmp_path):
    elements=[station(),station(2,'node')]
    buses,branches=gap_network()
    first=run_audit(elements,BOUNDARY,buses,branches,tmp_path/'first')
    second=run_audit(elements[::-1],BOUNDARY,buses.iloc[::-1],branches.iloc[::-1],tmp_path/'second')
    assert first==second
    for path in (tmp_path/'first').glob('*.json'):
        assert json.loads(path.read_text())==json.loads((tmp_path/'second'/path.name).read_text())
    assert first['automatic_spatial_merges']==0
    assert first['independent_reference_supplied'] is False


def test_empty_input_produces_valid_report(tmp_path):
    report=run_audit([],BOUNDARY,pd.DataFrame(),pd.DataFrame(),tmp_path)
    assert report['duplicate_pairs']==report['gap_candidates']==0
    assert json.loads((tmp_path/'substation_duplicate_candidates.json').read_text())==[]


def test_duplicate_osm_identity_is_rejected():
    with pytest.raises(ValueError,match='Duplicate OSM identity'):
        collect_substations([station(),station()],BOUNDARY)
