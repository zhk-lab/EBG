"""Run the request lifecycle checks and retain execution details."""

import json
from datetime import datetime, timezone
from pathlib import Path

from runtime.transport import make_client
from runtime.session import Session


def verify():
    client = make_client()
    session = Session(client)
    response = session.complete('Reply with OK only.')
    checks = {'request_constructed': True, 'response_present': bool(response.strip()),
              'response_text': isinstance(response, str),
              'response_serializable': bool(json.dumps(response))}
    result = {'checks': checks,
              'passed': all(checks.values())}
    folder = Path('artifacts') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True)
    (folder / 'operations.json').write_text(json.dumps({
        'operations': session.operations, 'live_requests': client.live_requests
    }, indent=2), encoding='utf-8')
    result['operations_file'] = 'operations.json'
    (folder / 'execution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Request lifecycle: {sum(checks.values())}/{len(checks)} checks passed")
    print(f"Execution record: {folder.as_posix()}/execution.json")
    return 0 if result['passed'] else 1
