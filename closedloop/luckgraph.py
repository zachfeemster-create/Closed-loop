"""Exploratory connections between logged rotation observations."""
import hashlib
import json
from itertools import combinations


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build(runs, attribution=None):
    from .loglobe import outcome
    nodes = {}
    for run in runs:
        for step in run.get('log', []):
            state = outcome(step)
            if state == 'error':
                continue
            dataset = run.get('key', {}).get(step.get('dataset'), step.get('dataset', '?'))
            identity = [run['id'], dataset, step.get('lens'), step.get('eq'), run.get('curve', {}), step]
            cid = 'L-' + fingerprint(identity)
            nodes[cid] = dict(id=cid, run=str(run['id']), dataset=str(dataset),
                              lens=str(step.get('lens', '?')), eq=str(step.get('eq', '?')),
                              state=state, credit=(attribution or {}).get(str(run['id'])) or
                              (run.get('display_name') if run.get('attribution_opt_in') is True else None) or 'Anonymous',
                              validation='untested')
    ordered = sorted(nodes.values(), key=lambda n: n['id'])
    edges = []
    # Explicit metadata matches only. Shared labels never establish a scientific relationship.
    for a, b in combinations(ordered, 2):
        if a['run'] == b['run']:
            continue
        reasons = []
        if a['dataset'] == b['dataset']:
            reasons.append('same dataset label across runs')
        if a['lens'] not in ('?', 'None', '') and a['eq'] not in ('?', 'None', '') and a['lens'] == b['lens'] and a['eq'] == b['eq']:
            reasons.append('same lens and EQ label')
        if reasons:
            edges.append(dict(a=a['id'], b=b['id'], reasons=reasons, validation='untested'))
    degree = {n['id']: 0 for n in ordered}
    adjacency = {n['id']: set() for n in ordered}
    for edge in edges:
        degree[edge['a']] += 1
        degree[edge['b']] += 1
        adjacency[edge['a']].add(edge['b'])
        adjacency[edge['b']].add(edge['a'])
    # A bridge observation is an articulation point: removing it splits its component.
    def branches_without(cid):
        remaining = set(adjacency[cid])
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            seen = {cid}
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                remaining.discard(current)
                stack.extend(adjacency[current] - seen)
        return count
    for node in ordered:
        node['connections'] = degree[node['id']]
        node['status'] = 'echo' if degree[node['id']] else 'isolated'
        branches = branches_without(node['id'])
        if branches > 1:
            node['status'] = 'bridge'
        node['branches_joined'] = branches if branches > 1 else 0
    return dict(nodes=ordered, edges=edges, kind='exploratory metadata connections')


def show(runs):
    import streamlit as st
    st.subheader('Luck connections')
    with st.expander('Optional connection credit'):
        st.caption('Names are self-reported. Choose only a run you contributed. '
                   'Credit stays in this session and the downloaded map.')
        run_id = st.selectbox('Your contributed run', sorted({str(r['id']) for r in runs}),
                             key='luck_credit_run') if runs else None
        display_name = st.text_input('Display name', max_chars=60, key='luck_credit_name')
        consent = st.checkbox('Include this name with my selected run in this map', key='luck_credit_consent')
        if st.button('Apply credit', disabled=not (run_id and consent and display_name.strip())):
            st.session_state.setdefault('luck_attribution', {})[run_id] = display_name.strip()
        if st.button('Remove credit', disabled=not run_id):
            st.session_state.setdefault('luck_attribution', {}).pop(run_id, None)
    graph = build(runs, st.session_state.get('luck_attribution', {}))
    st.caption('Explore luck, passes, and misses across loaded logs. Links show matching metadata. '
               'Repeated observations and shared settings do not establish validation or causation.')
    if not graph['nodes']:
        st.info('Load rotation logs to explore connections.')
        return
    st.write(f"{len(graph['nodes'])} observations · {len(graph['edges'])} connections")
    # Bound the drawing while keeping the complete map available for export.
    visible = graph['nodes'][:150]
    ids = {n['id'] for n in visible}
    dot = ['graph luck {']
    for node in visible:
        label = f"{node['dataset']} | {node['state']} | {node['credit']}"
        dot.append(f"{json.dumps(node['id'])} [label={json.dumps(label)}];")
    for edge in graph['edges']:
        if edge['a'] in ids and edge['b'] in ids:
            dot.append(f"{json.dumps(edge['a'])} -- {json.dumps(edge['b'])};")
    dot.append('}')
    st.graphviz_chart('\n'.join(dot))
    if len(graph['nodes']) > len(visible):
        st.caption('Drawing shows the first 150 observations. The table and export include all observations.')
    bridges = [n for n in graph['nodes'] if n['status'] == 'bridge']
    for node in bridges:
        st.info(f"Bridge: {node['dataset']} joins {node['branches_joined']} groups · "
                f"{node['credit']} · run {node['run']} · {node['id']}")
    import pandas as pd
    st.dataframe(pd.DataFrame(graph['nodes']), hide_index=True, width='stretch')
    selected = st.selectbox('Inspect an observation', graph['nodes'],
                            format_func=lambda n: f"{n['id']} · {n['dataset']} · {n['state']}")
    links = [e for e in graph['edges'] if selected['id'] in (e['a'], e['b'])]
    for edge in links:
        other_id = edge['b'] if edge['a'] == selected['id'] else edge['a']
        other = next(n for n in graph['nodes'] if n['id'] == other_id)
        st.write(f"{other['dataset']} · {other['state']} · {other['credit']} · run {other['run']}")
        st.caption('; '.join(edge['reasons']))
    if not links:
        st.caption('No matching observation in the currently loaded logs.')
    st.download_button('Download connection map', json.dumps(graph, indent=2),
                       'luck_connections.json', 'application/json')