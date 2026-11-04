"""Automatic mode of the Sec. IV.B.3 review (resolve_unmatched.py --auto)."""
import json

import numpy as np
import pandas as pd
import pytest

import resolve_unmatched as review
from pipeline_integrity import manual_bus

# Along latitude 30, 0.01 degrees of longitude is about 0.96 km.
A, B, C = (-95.0, 30.0), (-94.9, 30.0), (-94.95, 30.05)


@pytest.fixture
def unattended(tmp_path, monkeypatch):
    """No map, no browser, and any attempt to prompt a person fails the test."""
    monkeypatch.setattr(review, 'WORK_DIR', tmp_path)
    monkeypatch.setattr(review, 'region', 'test')
    monkeypatch.setattr(review, '_build_folium_map', lambda *a: 'test.html')
    monkeypatch.setattr(review.webbrowser, 'open', lambda *a: None)

    def refuse(*a):
        raise AssertionError('automatic mode must not prompt')

    monkeypatch.setattr('builtins.input', refuse)
    return tmp_path


def buses(*specs):
    """specs: (lon, lat, kV); bus_i counts from 1."""
    rows = [manual_bus(i, f'S{i}', (lon, lat), kv, []) for i, (lon, lat, kv) in enumerate(specs, 1)]
    return (pd.DataFrame(rows), [(lon, lat) for lon, lat, _ in specs],
            [str(int(kv * 1000)) for _, _, kv in specs])


def lines(*specs):
    """specs: (osm_id, fbus, tbus, coords, kV)."""
    rows, geom = [], {}
    for osm_id, f, t, coords, kv in specs:
        geom[osm_id] = coords
        rows.append(dict(fbus=f if f else np.nan, tbus=t if t else np.nan, osm_id=osm_id,
                         voltage_V=kv * 1000, length_km=review._line_length_km(coords),
                         fbus_name=None, tbus_name=None))
    return pd.DataFrame(rows), geom


def run(line_specs, bus_specs, log=None):
    df, geom = lines(*line_specs)
    bus_df, coords, volts = buses(*bus_specs)
    log = log or review.ReviewLog(mode='automatic')
    out, out_buses = review.run_pass1(df, geom, bus_df, coords, volts, log=log)
    return out, out_buses, log


def rules(log):
    return [e['rule'] for e in log.entries if e.get('context') and e['rule'] != 'deferred']


def test_an_endpoint_at_a_substation_is_assigned_to_it(unattended):
    out, _, log = run([('way/1', 1, None, [A, (-94.9004, 30.0)], 138)],
                      [(*A, 138), (*B, 138)])
    assert out.tbus.tolist() == [2]
    assert rules(log) == ['at substation']


def test_two_open_ends_that_meet_are_one_line(unattended):
    """The merge outranks the 4.8 km proximity match either piece could take."""
    out, _, log = run([('way/1', 1, None, [A, (-94.95, 30.0)], 138),
                       ('way/2', None, 2, [(-94.95, 30.0), B], 138)],
                      [(*A, 138), (*B, 138)])
    assert len(out) == 1
    assert (out.fbus[0], out.tbus[0]) == (1, 2)
    assert out.length_km[0] == pytest.approx(review._line_length_km([A, B]), rel=1e-3)
    assert rules(log) == ['continues']


def test_an_end_on_another_line_taps_it_at_a_new_junction(unattended):
    host = [A, (-94.97, 30.0), (-94.93, 30.0), B]
    out, out_buses, log = run([('way/1', 1, 2, host, 138),
                               ('way/2', 3, None, [C, (-94.95, 30.0)], 138)],
                              [(*A, 138), (*B, 138), (*C, 138)])
    junction = out_buses.loc[out_buses.name.str.startswith('Junction')]
    assert len(junction) == 1 and junction.base_kV.iloc[0] == 138000
    j = int(junction.bus_i.iloc[0])
    pairs = sorted(tuple(sorted((int(f), int(t)))) for f, t in zip(out.fbus, out.tbus))
    assert pairs == sorted([(1, j), (2, j), (3, j)])
    halves = out.loc[out.osm_id.str.startswith('way/1#')]
    assert halves.length_km.sum() == pytest.approx(review._line_length_km(host), rel=1e-3)
    assert rules(log) == ['tee']


def test_a_long_dead_end_gets_a_substation_and_a_short_one_is_deleted(unattended):
    far = (-90.0, 30.0)
    out, out_buses, log = run(
        [('way/long', 1, None, [A, (-94.7, 30.0)], 138),               # about 29 km
         ('way/short', 2, None, [(-89.97, 30.0), (-89.96, 30.0)], 138)],  # 1 km, 3 km out
        [(*A, 138), (*far, 138)])
    assert out.osm_id.tolist() == ['way/long']
    inferred = out_buses.loc[out_buses.bus_i == out.tbus[0]]
    assert inferred.name.iloc[0].startswith('Inferred substation')
    assert sorted(rules(log)) == ['inferred substation', 'stub']


def test_a_line_returning_to_its_own_substation_is_deleted(unattended):
    out, _, log = run([('way/1', 1, None, [A, (-94.98, 30.01), (-94.99, 30.0)], 138)],
                      [(*A, 138), (-80.0, 30.0, 138)])
    assert out.empty
    assert rules(log) == ['self-loop']


def test_a_fragment_attached_to_nothing_is_deleted(unattended):
    out, _, log = run([('way/1', None, None, [(-80.0, 30.0), (-79.9, 30.0)], 138)],
                      [(*A, 138)])
    assert out.empty
    assert rules(log) == ['fragment']


def test_a_close_substation_at_another_voltage_takes_the_line_across_the_guard(unattended):
    out, _, log = run([('way/1', 1, None, [A, (-94.8, 30.0)], 138)],
                      [(*A, 138), (-94.79, 30.0, 345)])
    assert out.tbus.tolist() == [2]
    assert rules(log) == ['other-voltage substation']
    # The override is answered and recorded exactly as a reviewer's would be.
    assert any('Assign anyway' in e['prompt'] and e['answer'] == 'y' for e in log.entries)


def test_automatic_journal_replays_without_prompting(unattended):
    host = [A, (-94.97, 30.0), (-94.93, 30.0), B]
    specs = [('way/1', 1, 2, host, 138),
             ('way/2', 3, None, [C, (-94.95, 30.0)], 138),
             ('way/3', 1, None, [A, (-95.0, 29.8)], 138),
             ('way/4', None, None, [(-95.0, 29.8), (-95.0, 29.6)], 138)]
    bus_specs = [(*A, 138), (*B, 138), (*C, 138)]
    path = unattended / 'journal.json'
    first, first_buses, log = run(specs, bus_specs, review.ReviewLog(path, mode='automatic'))
    log.save('test')
    assert json.loads(path.read_text())['mode'] == 'automatic'

    replay = review.ReviewLog(path, replay=True)
    assert replay.mode == 'automatic'
    second, second_buses, _ = run(specs, bus_specs, replay)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_buses, second_buses)


def test_a_truncated_automatic_journal_stops_instead_of_prompting(unattended):
    path = unattended / 'journal.json'
    path.write_text(json.dumps(dict(mode='automatic', answers=[])))
    with pytest.raises(RuntimeError, match='no answer'):
        run([('way/1', 1, None, [A, (-94.9004, 30.0)], 138)],
            [(*A, 138), (*B, 138)], review.ReviewLog(path, replay=True))


# --------------------------------------------------- edited journals

def record(path, specs, bus_specs):
    """Run automatic review, save its journal, and return the saved answers."""
    run(specs, bus_specs, review.ReviewLog(path, mode='automatic'))[2].save('test')
    return json.loads(path.read_text())


def edit(path, journal, number, answer, drop=0):
    """Replace answer `number` (counted from 1), removing `drop` answers after it."""
    answers = journal['answers']
    answers[number - 1] = dict(answers[number - 1], answer=answer)
    del answers[number:number + drop]
    path.write_text(json.dumps(journal))


def number_of(journal, answer):
    return next(n for n, e in enumerate(journal['answers'], 1) if e['answer'] == answer)


FAR = (-90.0, 30.0)
DEAD_END = ('way/long', 1, None, [A, (-94.7, 30.0)], 138)     # gets a new substation
STUB = ('way/short', 2, None, [(-89.97, 30.0), (-89.96, 30.0)], 138)   # deleted after it


def test_a_follow_up_answer_reaching_another_endpoint_stops_the_replay(unattended):
    """`new` swapped for a bus number leaves its name, kV and secondary behind."""
    path = unattended / 'journal.json'
    journal = record(path, [DEAD_END, STUB], [(*A, 138), (*FAR, 138)])
    n = number_of(journal, 'new')
    edit(path, journal, n, '2')
    with pytest.raises(RuntimeError, match=rf'answer {n + 1} .*New substation name.*'
                                           r'has reached tbus of way/short'):
        run([DEAD_END, STUB], [(*A, 138), (*FAR, 138)], review.ReviewLog(path, replay=True))


def test_an_answer_recorded_for_another_endpoint_stops_the_replay(unattended):
    """A `delete` turned into an assignment brings the line's other end up."""
    loop = ('way/1', None, None, [A, (-94.997, 30.002), (-94.999, 30.0)], 138)
    tail = ('way/2', 1, None, [A, (-94.9004, 30.0)], 138)
    bus_specs = [(*A, 138), (*B, 138)]
    path = unattended / 'journal.json'
    journal = record(path, [loop, tail], bus_specs)
    assert [e['answer'] for e in journal['answers']] == ['delete', '2']
    edit(path, journal, 1, '1')
    with pytest.raises(RuntimeError, match=r'answer 2 .*recorded for tbus of way/2, '
                                           r'but the review has reached tbus of way/1'):
        run([loop, tail], bus_specs, review.ReviewLog(path, replay=True))


def test_unused_answers_fail_the_replay_and_the_journal_is_kept(unattended):
    path = unattended / 'journal.json'
    journal = record(path, [DEAD_END], [(*A, 138), (*FAR, 138)])
    edit(path, journal, number_of(journal, 'new'), '2')
    edited = path.read_text()

    log = review.ReviewLog(path, replay=True)
    run([DEAD_END], [(*A, 138), (*FAR, 138)], log)
    log.save('test')
    assert path.read_text() == edited       # not cut down to the answers used
    with pytest.raises(RuntimeError, match=r'3 journal answer\(s\) unused'):
        log.finish()


def test_a_consistent_correction_replays_cleanly(unattended):
    """One bus for another, with the same prompts, is the safe kind of edit."""
    specs = [('way/1', 1, None, [A, (-94.9004, 30.0)], 138)]
    bus_specs = [(*A, 138), (*B, 138), (-94.9, 30.003, 138)]
    path = unattended / 'journal.json'
    journal = record(path, specs, bus_specs)
    assert [e['answer'] for e in journal['answers']] == ['2']
    edit(path, journal, 1, '3')
    log = review.ReviewLog(path, replay=True)
    out, _, _ = run(specs, bus_specs, log)
    log.finish()
    assert out.tbus.tolist() == [3]


def test_auto_and_replay_cannot_be_combined():
    with pytest.raises(SystemExit):
        review.main(['--auto', '--replay'])


# ------------------------------------------------------------ geometry

def test_find_tap_locates_the_foot_on_the_nearest_segment():
    tee = review.find_tap((-94.95, 30.0005), 138000,
                          [('host', [A, (-94.93, 30.0), B], 138000)], 1.0)
    assert tee['key'] == 'host' and tee['segment'] == 0
    assert tee['point'] == pytest.approx((-94.95, 30.0))
    assert tee['km'] == pytest.approx(0.055, abs=0.005)


def test_find_tap_ignores_the_other_lines_ends_and_other_voltages():
    near_end = (-94.9985, 30.0)            # 0.14 km from the host's start
    assert review.find_tap(near_end, 138000, [('host', [A, B], 138000)], 1.0) is None
    assert review.find_tap((-94.95, 30.0), 138000, [('host', [A, B], 345000)], 1.0) is None


def test_split_line_shares_the_point_between_both_halves():
    coords = [A, (-94.95, 30.0), B]
    head, tail = review.split_line(coords, 1, 0.5, (-94.925, 30.0))
    assert head == [A, (-94.95, 30.0), (-94.925, 30.0)]
    assert tail == [(-94.925, 30.0), B]
    head, tail = review.split_line(coords, 1, 0.0, (-94.95, 30.0))
    assert head[-1] == tail[0] == (-94.95, 30.0)
