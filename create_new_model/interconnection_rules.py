"""Reusable OSM connection-state and synchronous-system classification rules."""

def mixed_interconnection_circuits(tags):
    systems = {s.strip().upper() for s in str(tags.get('transmission_system', '')).split(';')}
    return 'ERCOT' in systems and bool(systems & {'SPP', 'SWPP', 'MISO'})


def line_interconnection(tags):
    systems = {s.strip().upper() for s in str(tags.get('transmission_system', '')).split(';')}
    groups = ({'Texas'} if 'ERCOT' in systems else set()) | ({'Eastern'} if systems & {'SPP', 'SWPP', 'MISO'} else set())
    return next(iter(groups)) if len(groups) == 1 else None


def normally_open(tags):
    return (str(tags.get('switch', '')).lower() == 'open'
            or str(tags.get('operational_status', '')).lower() == 'normally_open'
            or 'normally open' in str(tags.get('note', '')).lower())
