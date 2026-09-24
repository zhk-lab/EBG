from pathlib import Path
import unittest
from unittest.mock import patch

from benchmarks.feedbacktrace.scripts import build_feedbacktrace as builder


class ReleaseScopeTests(unittest.TestCase):
    def test_pack_defaults_and_resume_paths_are_available_without_credentials(self):
        args = builder.build_parser().parse_args([
            'pack', '--provenance', 'source.json', '--source-dir', 'source',
            '--resume-decisions', 'saved.jsonl', '--queue-dir', 'queue',
        ])
        self.assertEqual((args.count, args.positive_count, args.target_count), (100, 100, 100))
        self.assertEqual(args.resume_decisions, Path('saved.jsonl'))
        self.assertEqual(args.provenance, Path('source.json'))
        with patch.object(builder, '_pack_MODEL', ''):
            with self.assertRaisesRegex(builder.BuildError, 'MODEL_NAME'):
                builder._pack_validate_model_settings()

    def test_accepted_annotation_requires_key_fields(self):
        annotation = {
            'sample_id': 'case', 'selection_status': 'accepted', 'verdict': 'KEY',
            'gold_evidence_ids': ['e1'], 'gold_verification_point': 'The Agent changed the boundary.',
            'criticality': 'must_disclose', 'reason_code': 'accepted_key',
        }
        selector = {'selection_status': 'eligible', 'verdict': 'KEY'}
        events = [{'evidence_id': 'e1', 'event_type': 'assistant_response', 'content': 'Changed the boundary.'}]
        errors = builder._pack_validate_annotation(annotation, sample_id='case', selector=selector,
                                                   local_events=events, long_events=events)
        self.assertEqual(errors, [])
        annotation['verdict'] = None
        errors = builder._pack_validate_annotation(annotation, sample_id='case', selector=selector,
                                                   local_events=events, long_events=events)
        self.assertIn('accepted_without_valid_verdict', errors)

    def test_pack_count_cannot_request_a_different_composition(self):
        builder._pack_validate_count_configuration(count=3, positive_count=3)
        with self.assertRaisesRegex(ValueError, 'KEY'):
            builder._pack_validate_count_configuration(count=3, positive_count=2)
