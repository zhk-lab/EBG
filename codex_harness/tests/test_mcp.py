from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
import tomllib

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.support import PROJECT_ROOT, ProjectTemporaryDirectory
from codex_harness.integration import write_integration
from codex_harness import Harness
from codex_harness.hooks.lifecycle import handle_hook


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_only_flow_over_mcp_without_recorded_prompts(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            demand = 'app.py::run must return 2.'
            (repo / 'BEG.md').write_text(demand, encoding='utf-8')
            (repo / 'app.py').write_text('def run():\n    return 1\n', encoding='utf-8')
            server = StdioServerParameters(command=sys.executable,
                args=['-m', 'codex_harness', '--state-dir', str(root / 'state'), 'serve'], cwd=str(PROJECT_ROOT))
            async with asyncio.timeout(45):
                async with stdio_client(server) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        async def call(name, args):
                            result = await session.call_tool(name, args)
                            self.assertFalse(result.isError, result.content)
                            return yaml.safe_load(result.content[0].text)
                        listing = await call('beg_list_task_sources', {'repo_path': str(repo)})
                        self.assertEqual(listing['prompts'], [])
                        plan = listing['plans'][0]['id']
                        selected = await call('beg_select_task', {'plan_ids': [plan]})
                        result = await call('beg_build_evidence_groups', {'task_id': selected['task_id'],
                            'requirements': [{'id': 'R1', 'check': demand,
                                              'refs': [{'source_id': plan, 'quote': demand}]}]})
                        self.assertIn('return 1', str(result['evidence_groups']['R1']['actual']['repo']))
                        self.assertNotIn('changes', result)
                        empty = await session.call_tool('beg_select_task', {})
                        self.assertTrue(empty.isError)

    async def test_three_tools_complete_recorded_task_and_continue_frozen_reads(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            code = repo / 'app.py'
            code.write_text('def run():\n    return 1\n', encoding='utf-8', newline='')
            harness = Harness(root / 'state')
            base = {'session_id': 'recorded', 'turn_id': 't1', 'cwd': str(repo)}
            prompt = 'Change app.py run to return 2.'
            handle_hook(harness, {**base, 'hook_event_name': 'UserPromptSubmit', 'prompt': prompt})
            code.write_text('def run():\n    return 2\n', encoding='utf-8', newline='')
            handle_hook(harness, {**base, 'hook_event_name': 'Stop', 'last_assistant_message': 'app.py run returns 2.'})
            handle_hook(harness, {**base, 'turn_id': 'review', 'hook_event_name': 'UserPromptSubmit', 'prompt': 'Disclose the last task.'})
            code.write_text('UNRELATED LIVE CODE', encoding='utf-8')
            server = StdioServerParameters(command=sys.executable, args=['-m', 'codex_harness', '--state-dir', str(root / 'state'), 'serve'], cwd=str(PROJECT_ROOT))
            async with asyncio.timeout(45):
                async with stdio_client(server) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        names = {t.name for t in (await session.list_tools()).tools}
                        self.assertEqual(names, {'beg_list_task_sources', 'beg_select_task', 'beg_build_evidence_groups'})
                        async def call(name, arguments):
                            response = await session.call_tool(name, arguments)
                            self.assertFalse(response.isError, response.content)
                            return yaml.safe_load(response.content[0].text)
                        listing = await call('beg_list_task_sources', {})
                        self.assertEqual(listing['prompts'][0]['content'], prompt)
                        selected = await call('beg_select_task', {'start_prompt': 'P1', 'end_prompt': 'P1', 'plan_ids': []})
                        task = selected['task_id']
                        invalid = await session.call_tool('beg_build_evidence_groups', {'task_id': task})
                        self.assertTrue(invalid.isError)
                        result = await call('beg_build_evidence_groups', {'task_id': task, 'requirements': [
                            {'id': 'R1', 'check': prompt, 'refs': [{'source_id': 'P1', 'quote': prompt}]}]})
                        self.assertIn('R1', result['evidence_groups'])
                        changes = await call('beg_build_evidence_groups', {'task_id': task, 'read_ref': result['changes']['read_ref']})
                        ref = next(e['read_ref'] for e in changes['content'] if e['path'] == 'app.py')
                        diff = await call('beg_build_evidence_groups', {'task_id': task, 'read_ref': ref})
                        self.assertIn('-    return 1', diff['content'])
                        self.assertIn('+    return 2', diff['content'])
                        self.assertNotIn('UNRELATED', diff['content'])
                        self.assertEqual(await call('beg_build_evidence_groups', {'task_id': task, 'read_ref': result['read_ref']}), result)

    def test_hook_cli_stdout_is_json_and_does_not_block_turn(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / "repo"
            repo.mkdir()
            prompt = "完成任务，核对中文要求。"
            result = subprocess.run(
                [sys.executable, "-m", "codex_harness", "--state-dir", str(root / "state"), "hook"],
                input=json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s", "turn_id": "t", "cwd": str(repo), "prompt": prompt}, ensure_ascii=False),
                env={**os.environ, "PYTHONIOENCODING": "gbk:surrogateescape", "PYTHONUTF8": "0"},
                text=True, encoding="utf-8", capture_output=True, cwd=PROJECT_ROOT, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertNotIn("decision", payload)
            self.assertIn("additionalContext", payload["hookSpecificOutput"])
            self.assertEqual(Harness(root / "state").sessions.prompts("s")[0]['content'], prompt)

    @unittest.skipUnless(os.name == "nt", "Windows command hook quoting")
    def test_generated_windows_hook_runs_in_powershell(self):
        with ProjectTemporaryDirectory() as root:
            output, state = root / "integration files", root / "state files ' $data"
            write_integration(output, state, sys.executable)
            config = tomllib.loads((output / "config.toml").read_text(encoding="utf-8"))
            self.assertIn(str(state), config["mcp_servers"]["beg_disclose"]["args"])
            hooks = json.loads((output / "hooks.json").read_text(encoding="utf-8"))
            command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["commandWindows"]
            # Codex uses the session shell; the Windows default is PowerShell.
            result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                    input=json.dumps({"session_id": "unknown", "hook_event_name": "SessionStart"}),
                                    capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('additionalContext', json.loads(result.stdout)['hookSpecificOutput'])
