"""Client for the separately deployed pair-scoring service."""
import json
from pathlib import Path
import urllib.request


class PairModel:
    def __init__(self):
        self.options = json.loads(Path('service.json').read_text())

    def score(self, candidate):
        payload = {'model': self.options['model'], 'features': candidate['pair_features']}
        request = urllib.request.Request(
            self.options['url'].rstrip('/') + '/score',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            return float(json.load(response)['score'])
