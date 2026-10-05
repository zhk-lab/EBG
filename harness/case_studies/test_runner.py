"""Runner failure and environment regressions; no model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import run as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspaces/r1/plain/01_ambiguity'
        self.workspace.mkdir(parents=True)
        self.output = self.root / 'runs/r1/plain/01_ambiguity'
        self.options = dict(plain=True, workspace_root=self.root / 'workspaces',
                            output_root=self.root / 'runs')
        self.help = subprocess.CompletedProcess([], 0, '--ignore-user-config', '')

    def run_case(self):
        return runner.run('01_ambiguity', 'r1', **self.options)

    def write_status(self, status):
        self.output.mkdir(parents=True)
        (self.output / 'status.json').write_text(json.dumps(status), encoding='utf-8')

    def status(self):
        return json.loads((self.output / 'status.json').read_text(encoding='utf-8'))

    def test_cached_failure_keeps_nonzero_exit_without_restarting(self):
        self.write_status({'exit_code': 7})
        with patch.object(runner.subprocess, 'Popen') as popen:
            self.assertEqual(self.run_case(), 7)
        popen.assert_not_called()

    def test_legacy_success_is_checked_against_the_actual_turn(self):
        self.write_status({'exit_code': 0})
        (self.output / 'events.jsonl').write_text('{"type":"turn.failed","error":"failed"}\n', encoding='utf-8')
        self.assertEqual(self.run_case(), 1)
        self.assertEqual(self.status()['exit_code'], 1)

    def test_legacy_completed_turn_can_be_reused(self):
        self.write_status({'exit_code': 0})
        (self.output / 'events.jsonl').write_text('{"type":"turn.completed","usage":{}}\n', encoding='utf-8')
        self.assertEqual(self.run_case(), 0)
        self.assertTrue(self.status()['turn_completed'])

    def test_relative_cli_is_resolved_before_the_case_changes_directory(self):
        with patch.object(runner.shutil, 'which', return_value='bin/codex.exe'):
            self.assertEqual(runner.resolve_codex('bin/codex.exe'), str(Path('bin/codex.exe').resolve()))

    @unittest.skipUnless(os.name == 'nt', 'npm Windows wrapper')
    def test_npm_cmd_wrapper_uses_native_executable(self):
        native = self.root / 'node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/codex/codex.exe'
        native.parent.mkdir(parents=True)
        native.touch()
        with patch.object(runner.shutil, 'which', return_value=str(self.root / 'codex.CMD')), \
                patch.object(runner.platform, 'machine', return_value='AMD64'):
            self.assertEqual(runner.resolve_codex('codex'), str(native))

    @unittest.skipUnless(os.name == 'nt', 'npm Windows wrapper')
    def test_npm_sibling_package_with_bin_layout_uses_native_executable(self):
        native = self.root / 'node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe'
        native.parent.mkdir(parents=True)
        native.touch()
        with patch.object(runner.shutil, 'which', return_value=str(self.root / 'codex.CMD')), \
                patch.object(runner.platform, 'machine', return_value='AMD64'):
            self.assertEqual(runner.resolve_codex('codex'), str(native))

    @unittest.skipUnless(os.name == 'nt', 'npm Windows wrapper')
    def test_local_npm_bin_wrapper_uses_native_executable(self):
        native = self.root / 'node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe'
        native.parent.mkdir(parents=True)
        native.touch()
        with patch.object(runner.shutil, 'which', return_value=str(self.root / 'node_modules/.bin/codex.cmd')), \
                patch.object(runner.platform, 'machine', return_value='AMD64'):
            self.assertEqual(runner.resolve_codex('codex'), str(native))

    def test_unfinished_run_is_preserved(self):
        self.write_status({'exit_code': None, 'pid': 123})
        with self.assertRaisesRegex(RuntimeError, 'unfinished'):
            self.run_case()
        self.assertEqual(self.status()['pid'], 123)

    def test_incompatible_cli_fails_before_creating_run_state(self):
        result = subprocess.CompletedProcess([], 2, '', 'unsupported option')
        with patch.object(runner.shutil, 'which', return_value=sys.executable), \
                patch.object(runner.subprocess, 'run', return_value=result), \
                patch.object(runner.subprocess, 'Popen', side_effect=OSError('must not launch')) as popen:
            with self.assertRaisesRegex(RuntimeError, 'Update'):
                self.run_case()
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_spawn_failure_is_a_finished_failed_run(self):
        with patch.object(runner.shutil, 'which', return_value=sys.executable), \
                patch.object(runner.subprocess, 'run', return_value=self.help), \
                patch.object(runner.subprocess, 'Popen', side_effect=OSError('launch failed')):
            self.assertEqual(self.run_case(), 1)
        self.assertEqual(self.status()['exit_code'], 1)
        self.assertEqual(self.status()['error'], 'launch failed')

    def launch_with_events(self, events):
        process = Mock(pid=123, returncode=0)

        def launch(*args, **kwargs):
            self.spawn_env = kwargs['env']
            kwargs['stdout'].write('\n'.join(json.dumps(event) for event in events) + '\n')
            return process

        with patch.object(runner.shutil, 'which', return_value=sys.executable), \
                patch.object(runner.subprocess, 'run', return_value=self.help), \
                patch.object(runner.subprocess, 'Popen', side_effect=launch):
            return self.run_case()

    def test_zero_process_exit_without_completed_turn_is_failure(self):
        self.assertEqual(self.launch_with_events([{'type': 'thread.started', 'thread_id': 't1'}]), 1)
        self.assertFalse(self.status()['turn_completed'])

    def test_failed_turn_is_not_reported_as_success(self):
        self.assertEqual(self.launch_with_events([{'type': 'turn.failed', 'error': {'message': 'rate limit'}}]), 1)
        self.assertEqual(self.status()['error']['message'], 'rate limit')

    def test_recovered_stream_error_does_not_invalidate_completed_turn(self):
        self.assertEqual(self.launch_with_events([
            {'type': 'error', 'message': 'stream disconnected; retrying'},
            {'type': 'turn.completed', 'usage': {}},
        ]), 0)
        self.assertIn('stream disconnected', self.status()['diagnostics'][0])

    def test_completed_turn_retains_usage_and_uses_current_python_environment(self):
        self.assertEqual(self.launch_with_events([
            {'type': 'thread.started', 'thread_id': 't1'},
            {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 5}},
        ]), 0)
        status = self.status()
        self.assertTrue(status['turn_completed'])
        self.assertEqual(status['thread_id'], 't1')
        self.assertEqual(status['usage']['output_tokens'], 5)
        self.assertEqual(self.spawn_env['PATH'].split(os.pathsep)[0], str(Path(sys.executable).parent))


if __name__ == '__main__':
    unittest.main()
