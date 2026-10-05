"""Run an isolated Luna-low session, preserving raw events and completion state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tomllib

from prepare import CASES, WORKSPACES, ROOT, STUDY


def toml(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + '=' + toml(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(toml(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False)


def resolve_codex(codex):
    cli = shutil.which(codex)
    if cli is None:
        raise FileNotFoundError('Codex CLI not found; install it or pass --codex /path/to/codex')
    cli = Path(cli).resolve()
    # Avoid cmd.exe rewriting the inline TOML arguments passed through npm's wrapper.
    if os.name == 'nt' and cli.suffix.lower() in ('.cmd', '.bat'):
        architecture = 'aarch64' if platform.machine().lower() in ('arm64', 'aarch64') else 'x86_64'
        modules = cli.parent.parent if cli.parent.name == '.bin' else cli.parent / 'node_modules'
        packages = (modules / '@openai/codex/node_modules/@openai', modules / '@openai')
        for directory in packages:
            candidates = [path for layout in ('bin', 'codex') for path in directory.glob(
                f'codex-win32-*/vendor/{architecture}-pc-windows-msvc/{layout}/codex.exe')]
            if len(candidates) == 1:
                cli = candidates[0]
                break
    return str(cli)


def check_cli(cli, plain):
    command = [cli, 'exec']
    if not plain:
        command.append('--dangerously-bypass-hook-trust')
    result = subprocess.run(command + ['--help'], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=30)
    if result.returncode or '--ignore-user-config' not in result.stdout:
        raise RuntimeError('This Codex CLI lacks the required exec/hook support. '
                           'Update with npm install -g @openai/codex@latest, or use '
                           '--codex /path/to/a/newer/codex. No run was started.')


def summarize_events(path):
    summary = {'turn_completed': False}
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('type') == 'thread.started':
            summary['thread_id'] = event.get('thread_id')
        elif event.get('type') == 'turn.completed':
            summary['turn_completed'] = True
            summary.pop('error', None)
            summary['usage'] = event.get('usage', {})
        elif event.get('type') == 'turn.failed':
            summary['turn_completed'] = False
            summary['error'] = event.get('error', event.get('message', event))
        elif event.get('type') == 'error':
            summary.setdefault('diagnostics', []).append(event.get('message', event))
    return summary


def run(case, round_name, plain=False, *, workspace_root=WORKSPACES,
        output_root=ROOT / 'outputs/harness/runs', codex='codex', model='gpt-6-luna', effort='low'):
    arm = 'plain' if plain else 'harness'
    repo = (workspace_root / round_name / arm / case).resolve()
    output = (output_root / round_name / arm / case).resolve()
    status_path = output / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('exit_code') is not None:
            if status['exit_code'] == 0 and 'turn_completed' not in status:
                events_path = output / 'events.jsonl'
                status.update(summarize_events(events_path) if events_path.exists() else {'turn_completed': False})
                if not status['turn_completed']:
                    status['exit_code'] = 1
                    status.setdefault('error', 'Cached run has no completed turn; inspect its raw logs.')
                status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
            print(json.dumps(status))
            return status['exit_code']
        raise RuntimeError('An unfinished run exists; inspect its PID and raw logs before restarting. '
                           'Use a new --round for an independent retry.')
    if not repo.is_dir():
        raise FileNotFoundError(f'Prepare the case workspace first: {repo}')
    cli = resolve_codex(codex)
    check_cli(cli, plain)
    command = [cli, 'exec', '--ignore-user-config', '--skip-git-repo-check',
               '-C', str(repo), '-s', 'workspace-write', '-m', model,
               '-c', 'model_reasoning_effort=' + json.dumps(effort), '-c', 'features.multi_agent=false',
               '--json', '-o', str(output / 'answer.txt'),
               '-c', 'projects.' + json.dumps(str(repo)) + '.trust_level="trusted"']
    if os.name == 'nt':
        command += ['-c', 'windows.sandbox="unelevated"']
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
    output.mkdir(parents=True)
    shutil.copytree(STUDY.parent / 'src', output / 'harness-source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    status = {'case': case, 'round': round_name, 'arm': arm, 'model': model, 'effort': effort,
              'started': time.time(), 'exit_code': None, 'command': command}

    def save_status():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

    save_status()
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    with (output / 'events.jsonl').open('w', encoding='utf-8') as stdout, (output / 'stderr.log').open('w', encoding='utf-8') as stderr:
        try:
            process = subprocess.Popen(command + ['-'], cwd=repo, env=env, stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, text=True, encoding='utf-8')
        except OSError as error:
            status.update(exit_code=1, error=str(error), elapsed=time.time() - status['started'])
            save_status()
            print(json.dumps({k: v for k, v in status.items() if k != 'command'}))
            return 1
        status['pid'] = process.pid
        save_status()
        try:
            process.communicate(prompt, timeout=1200)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status['timeout'] = True
        status['process_exit_code'] = process.returncode
    status.update(summarize_events(output / 'events.jsonl'))
    status['exit_code'] = 124 if status.get('timeout') else process.returncode
    if status['exit_code'] == 0 and (not status['turn_completed'] or status.get('error')):
        status['exit_code'] = 1
        status.setdefault('error', 'Codex exited without a completed turn; inspect events.jsonl and stderr.log.')
    status['elapsed'] = time.time() - status['started']
    save_status()
    print(json.dumps({k: v for k, v in status.items() if k != 'command'}))
    return status['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--round', default='r1')
    parser.add_argument('--plain', action='store_true')
    parser.add_argument('--workspace-root', type=Path, default=WORKSPACES)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/harness/runs')
    parser.add_argument('--codex', default='codex')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--effort', default='low')
    args = parser.parse_args()
    sys.exit(run(args.case, args.round, plain=args.plain, workspace_root=args.workspace_root,
                 output_root=args.output_root, codex=args.codex, model=args.model, effort=args.effort))
