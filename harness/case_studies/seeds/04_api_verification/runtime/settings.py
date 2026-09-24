"""Service settings shared by experiment entry points."""

import os
from pathlib import Path


def load_environment(path=Path('.env')):
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def service_options():
    load_environment()
    return {'url': os.getenv('MODEL_BASE_URL', 'https://api.deepseek.com'),
            'model': os.getenv('MODEL_NAME', 'deepseek-chat'),
            'token': os.getenv('MODEL_API_KEY', '')}
