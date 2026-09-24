import tempfile
import unittest
from pathlib import Path

from attribute import numbered_sources, location_checks, classify, full_event_covered


class EvidenceAuditTests(unittest.TestCase):
    def test_cross_file_context_and_return_to_root(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)
            (repo/'a.py').write_text('a\n',encoding='utf-8')
            (repo/'b.py').write_text('b\n',encoding='utf-8')
            text='a.py::f@1\n1 | a\n[CONTEXT via calls]\nf@1 calls b.py::g@1\n1 | b\n[CONTEXT via feeds]\nf@1 feeds h@1\n1 | a'
            got=numbered_sources(text,repo)
            self.assertEqual(dict(got),{'a.py':{1},'b.py':{1}})

    def test_split_event_excerpts_cover_original(self):
        self.assertTrue(full_event_covered('claim\n- first\n- second', ['- second','claim','- first']))
        self.assertFalse(full_event_covered('claim\n- first\n- second', ['claim','- first']))

    def test_source_text_and_file_must_match_not_just_line_number(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo/'a.py').write_text('x = 1\ny = 2\n', encoding='utf-8')
            got = numbered_sources('FILE a.py\n1 | x = 1\n2 | invented\n', repo)
            self.assertEqual(got['a.py'], {1})

    def test_repeated_blocks_combine_but_gap_remains_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo/'a.py').write_text('a\nb\nc\n', encoding='utf-8')
            got = numbered_sources('a.py::f@1-3\n1 | a\n...\n3 | c', repo)
            loc = {'file':'a.py','line_ranges':[{'start':1,'end':3}]}
            checked = location_checks([loc],got,repo)[0]
            self.assertFalse(checked['complete'])
            self.assertEqual(checked['missing'],[2])

    def test_trace_ids_alone_do_not_prove_full_evidence(self):
        row = {'benchmark':'feedbacktrace','gold':{'evidence_ids':['e1']}}
        audit = {'trace_checks':{'e1':{'present':True,'in_current_scope':True,'complete':False}}}
        self.assertIsNone(classify(row,audit)[0])
        audit['trace_checks']['e1']['complete']=True
        self.assertEqual(classify(row,audit)[0],'model_judgment')

    def test_historical_evidence_is_not_allowed_current_evidence(self):
        row = {'benchmark':'feedbacktrace','gold':{'evidence_ids':['e1']}}
        audit = {'trace_checks':{'e1':{'present':True,'in_current_scope':False,'complete':False}}}
        self.assertEqual(classify(row,audit)[0],'evidence_presentation')


if __name__=='__main__':
    unittest.main()
