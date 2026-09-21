"""Run Step 3 unattended under one of two explicit policies.

Sec. IV.B.3 puts a human in the loop. Neither policy here is a substitute for
that review; both exist so the pipeline can be exercised end to end and so a
baseline is reproducible.

  skip      every endpoint is left unmatched and its line is dropped. Nothing is
            invented, but the network fragments into islands.
  assisted  each endpoint takes the nearest candidate the pipeline itself offers
            -- the same voltage-compatible, within-radius list the CLI prints for
            the reviewer -- and falls back to skip when no candidate qualifies.
            This is the pipeline's own ranking applied automatically, so it
            reproduces the easy decisions a reviewer would make and must still be
            checked before the result is called reviewed.

Usage: python auto_review.py [skip|assisted]
"""
import builtins
import sys

import resolve_unmatched as review

MAX_TRIES = 4          # candidates to try at one endpoint before giving up
state = {'queue': [], 'answers': 0, 'assigned': 0, 'skipped': 0}


def install(policy):
    original = review.nearby_candidates

    def wrapped(*args, **kwargs):
        found = original(*args, **kwargs)
        # A fresh candidate list means a new endpoint is being asked about.
        state['queue'] = [str(int(c[2])) for c in found[:MAX_TRIES]]
        return found

    def answer(prompt=''):
        state['answers'] += 1
        text = str(prompt)
        if 'Assign anyway' in text:
            return 'n'          # never override a voltage mismatch unattended
        if 'Choice (A/B)' in text:
            return 'A'
        if 'Add a' in text and 'line to join' in text:
            # Closing a gap OSM never mapped is a reviewer's call; the assisted
            # policy takes the pipeline's own proposal, skip takes none.
            return 'y' if policy == 'assisted' else 'skip'
        if policy == 'assisted' and state['queue']:
            state['assigned'] += 1
            return state['queue'].pop(0)
        state['skipped'] += 1
        return 'skip'

    review.nearby_candidates = wrapped
    builtins.input = answer


def main():
    policy = (sys.argv[1] if len(sys.argv) > 1 else 'skip').lower()
    if policy not in ('skip', 'assisted'):
        raise SystemExit('policy must be skip or assisted')
    install(policy)
    review.webbrowser.open = lambda *a, **k: None
    # The Folium map is a reviewer aid only, and slow on large cases.
    review._build_folium_map = lambda *a, **k: 'skipped.html'
    review.main([])
    print(f'AUTO-REVIEW[{policy}]: {state["answers"]} prompt(s), '
          f'{state["assigned"]} assignment(s), {state["skipped"]} skip(s)')


if __name__ == '__main__':
    main()
