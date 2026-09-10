"""Minimal model transport and offline development implementation."""

import json
import urllib.request

from runtime.settings import service_options


class ReplayClient:
    mode = 'mock'

    def __init__(self):
        self.live_requests = 0
        self.response_ids = []

    def complete(self, prompt):
        return 'OK'


class ServiceClient:
    mode = 'live'

    def __init__(self, options):
        self.options = options
        self.live_requests = 0
        self.response_ids = []

    def complete(self, prompt):
        payload = {'model': self.options['model'],
                   'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 32}
        request = urllib.request.Request(
            self.options['url'].rstrip('/') + '/chat/completions',
            data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + self.options['token'],
                     'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        text = result['choices'][0]['message']['content']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty model response')
        self.live_requests += 1
        self.response_ids.append(result.get('id'))
        return text


def make_client():
    options = service_options()
    return ServiceClient(options) if options['token'] else ReplayClient()
