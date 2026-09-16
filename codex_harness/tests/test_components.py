from __future__ import annotations

import unittest

from codex_harness.application.components import components


def node(name):
    return ('app.py', name)


def edge(left, right, kind='calls'):
    return {'from': dict(zip(('path', 'symbol'), node(left))),
            'to': dict(zip(('path', 'symbol'), node(right))),
            'type': kind, 'evidence_ids': []}


class ComponentTests(unittest.TestCase):
    def test_all_one_hop_neighbors_without_recursive_expansion(self):
        edges = [edge('start', f'neighbor{i}') for i in range(7)]
        edges.append(edge('neighbor0', 'outside'))
        result = components({node('start')}, edges, '`start`')
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['nodes']), 8)
        self.assertEqual(len(result[0]['edges']), 7)
        self.assertNotIn(node('outside'), result[0]['nodes'])

    def test_root_uses_distinct_neighbors_among_seeds_not_edge_count(self):
        edges = [edge('a', 'shared')] * 5 + [edge('a', 'a')]
        edges += [edge('b', 'shared'), edge('extra', 'b')]
        result = components({node('a'), node('b')}, edges, '`a` then `b`')
        self.assertEqual(result[0]['root'], node('b'))
        self.assertEqual(result[0]['nodes'][0], node('b'))

    def test_shared_neighbor_tie_cycle_isolate_and_document_order(self):
        edges = [edge('a', 'shared'), edge('b', 'shared'), edge('shared', 'a')]
        result = components({node('a'), node('b'), node('alone')}, edges, '`b` then `a` then `alone`')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['root'], node('b'))
        self.assertEqual(set(result[0]['nodes']), {node('a'), node('b'), node('shared')})
        self.assertEqual(result[1]['nodes'], [node('alone')])
        self.assertEqual(result[0]['edges'], edges)

    def test_tied_fallbacks_are_stable_when_inputs_are_reordered(self):
        edges = [edge('a', 'shared'), edge('b', 'shared')]
        seeds = {node('a'), node('b')}
        forward = components(seeds, edges, '')
        reverse = components(seeds, list(reversed(edges)), '')
        self.assertEqual(forward[0]['root'], node('a'))
        self.assertEqual(forward[0]['nodes'], reverse[0]['nodes'])

    def test_unqualified_method_mentions_break_ties_in_document_order(self):
        edges = [edge('Client.first', 'shared'), edge('Client.second', 'shared')]
        result = components({node('Client.first'), node('Client.second')}, edges, '`second` then `first`')
        self.assertEqual(result[0]['root'], node('Client.second'))
