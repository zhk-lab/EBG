"""Run a fresh Luna-low session; retain raw events and restart-safe completion state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib

from prepare import CASES, DESKTOP, STUDY


def toml(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + '=' + toml(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(toml(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False)


def run(case, round_name, plain=False):
    repo = DESKTOP / round_name / case
    output = STUDY.parent / '.tmp/harness-five-cases' / round_name / case
    status_path = output / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('exit_code') is not None:
            print(json.dumps(status))
            return
        raise RuntimeError('An unfinished run exists; inspect its PID and raw logs before restarting.')
    output.mkdir(parents=True, exist_ok=True)
    cli = max((Path(os.environ['LOCALAPPDATA']) / 'OpenAI/Codex/bin').glob('*/codex.exe'),
              key=lambda p: p.stat().st_mtime)
    command = [str(cli), 'exec', '--ignore-user-config', '--skip-git-repo-check',
               '-C', str(repo), '-s', 'workspace-write', '-m', 'gpt-5.6-luna',
               '-c', 'model_reasoning_effort="low"', '-c', 'features.multi_agent=false',
               '-c', 'windows.sandbox="unelevated"', '--json', '-o', str(output / 'answer.txt'),
               '-c', 'projects.' + json.dumps(str(repo)) + '.trust_level="trusted"']
    if plain:
        command += ['-c', 'features.hooks=false']
    else:
        config = tomllib.loads((repo / '.codex/config.toml').read_text(encoding='utf-8'))
        hooks = json.loads((repo / '.codex/hooks.json').read_text(encoding='utf-8'))['hooks']
        command += ['--dangerously-bypass-hook-trust', '-c', 'features.hooks=true',
                    '-c', 'mcp_servers=' + toml(config['mcp_servers']),
                    '-c', 'mcp_servers.ebg_disclose.enabled=true',
                    '-c', 'mcp_servers.ebg_disclose.required=true', '-c', 'hooks=' + toml(hooks)]
    if case == '04_api_verification':
        command += ['-c', 'sandbox_workspace_write.network_access=true']
    prompt = (STUDY / 'prompts' / (case + '.md')).read_text(encoding='utf-8')
    shutil.copytree(STUDY.parent / 'codex_harness/src', output / 'harness-source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    status = {'case': case, 'round': round_name, 'arm': 'plain' if plain else 'harness', 'model': 'gpt-5.6-luna', 'effort': 'low',
              'started': time.time(), 'exit_code': None, 'command': command}
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    with (output / 'events.jsonl').open('w', encoding='utf-8') as stdout, (output / 'stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen(command + ['-'], cwd=repo, env=env, stdin=subprocess.PIPE,
                                   stdout=stdout, stderr=stderr, text=True, encoding='utf-8')
        status['pid'] = process.pid
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        try:
            process.communicate(prompt, timeout=1200)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
            status['timeout'] = True
        status['exit_code'] = process.returncode
    status['elapsed'] = time.time() - status['started']
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in status.items() if k != 'command'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--round', default='r1')
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()
    run(args.case, args.round, plain=args.plain)
