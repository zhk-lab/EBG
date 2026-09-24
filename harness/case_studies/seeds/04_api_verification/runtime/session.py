"""Persist completed service operations so interrupted sessions can resume."""

import json
from pathlib import Path


class Session:
    def __init__(self, client, state=Path('runtime/state/completed.json')):
        self.client = client
        self.state = state
        self.completed = json.loads(state.read_text(encoding='utf-8')) if state.exists() else {}
        self.operations = []

    def complete(self, prompt):
        if prompt in self.completed:
            entry = self.completed[prompt]
            source = 'completed_state'
        else:
            text = self.client.complete(prompt)
            entry = {'text': text, 'response_ids': list(self.client.response_ids)}
            self.completed[prompt] = entry
            self.state.parent.mkdir(parents=True, exist_ok=True)
            self.state.write_text(json.dumps(self.completed, indent=2), encoding='utf-8')
            source = 'service'
        self.operations.append({'source': source, 'state_path': self.state.as_posix(),
                                'response_ids': entry['response_ids']})
        return entry['text']
