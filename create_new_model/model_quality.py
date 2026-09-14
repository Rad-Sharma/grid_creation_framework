"""Separate coverage, assumptions, and feasibility instead of one accuracy flag."""
from collections import Counter


def quality_report(buses, branches, omitted, generators=None, rejected=None):
    kinds = Counter(b.get('kind','unknown') for b in branches)
    return dict(
        topology=dict(buses=len(buses),branches=len(branches),branch_kinds=dict(kinds),
            station_connection_evidence=buses.get('connection_evidence', []).value_counts().to_dict()
                if 'connection_evidence' in buses else {},
            unresolved_items=int(len(omitted))),
        generation=dict(mapped_units=len(generators) if generators is not None else None,
            withheld_units=len(rejected) if rejected is not None else None),
        geographic_coverage=dict(independently_benchmarked=False,coverage_fraction=None),
        equipment=dict(independently_verified=False,
            reviewed_connections=sum(count for kind,count in kinds.items() if kind.startswith('reviewed_')),
            legacy_assumed_transformers=kinds.get('assumed_transformer',0),
            line_parameters='synthetic voltage/length fits, not verified equipment data'),
        operating_feasibility=dict(ac_validated=False,report='Run constrained_dispatch.py on the selected scenario'),
        interpretation='Fewer reported paths or passing software tests do not establish physical accuracy')
