"""Schedule fixed model requests while preserving result order."""


def execute(requests, client, completed=None):
    results = []
    for request in requests:
        results.append(client.complete(request))
    return results
