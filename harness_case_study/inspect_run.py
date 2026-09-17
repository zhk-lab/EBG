"""Compact inspection of raw agent events without dumping data or credentials."""
import argparse
import json
import sqlite3
from prepare import STUDY, DESKTOP

parser = argparse.ArgumentParser()
parser.add_argument('case')
parser.add_argument('--round', default='r1')
args = parser.parse_args()
folder = STUDY.parent / '.tmp/harness-five-cases' / args.round / args.case
events = []
for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines():
    try:
        events.append(json.loads(line))
    except ValueError:
        pass
for event in events:
    item = event.get('item', {})
    if event['type'] == 'item.completed' and item.get('type') == 'mcp_tool_call':
        print(json.dumps({'tool': item.get('tool'), 'arguments': item.get('arguments'),
                          'status': item.get('status'), 'error': item.get('error')}, ensure_ascii=False))
    elif event['type'] == 'item.completed' and item.get('type') == 'agent_message':
        print('MESSAGE:', item.get('text', '')[:1500])
db_path = DESKTOP / args.round / args.case / '.ebg-harness/harness.sqlite3'
if db_path.exists():
    with sqlite3.connect(db_path) as db:
        for (data,) in db.execute('SELECT data FROM checkpoints'):
            check = json.loads(data)
            print('CHECK:', json.dumps({k: check.get(k) for k in ('id','trigger','evidence_queries','assessment')}, ensure_ascii=False))
        print('EVENTS:', db.execute('SELECT count(*) FROM recorded_events').fetchone()[0])
if (folder / 'answer.txt').exists():
    print('FINAL:', (folder / 'answer.txt').read_text(encoding='utf-8'))
