"""Exact text slices backed by immutable file checkpoints."""

from __future__ import annotations

from typing import Any

from .storage import HarnessError, Store


class SourceReader:
    """Cache each file once during a query; never read the live workspace."""

    def __init__(self, store: Store):
        self.store = store
        self.files: dict[int, list[str]] = {}

    def read(self, ref: dict[str, Any]) -> str:
        file_id = ref['file_id']
        if file_id not in self.files:
            self.files[file_id] = self.store.file(file_id)['content'].splitlines(keepends=True)
        first, last = ref['lines']
        if not 1 <= first <= last <= max(1, len(self.files[file_id])):
            raise HarnessError('Source slice is outside the frozen file.')
        return ''.join(self.files[file_id][first - 1:last])

    def context(self, context: dict[str, Any]) -> str:
        return context['source'] if 'source' in context else self.read(context['source_ref'])


def referenced_context(context: dict[str, Any], files: dict[str, int]) -> dict[str, Any]:
    return {**{key: value for key, value in context.items() if key != 'source'},
            'source_ref': {'file_id': files[context['path']], 'lines': context['lines']}}


def pack_output(value: Any) -> Any:
    """Persist generated excerpt bodies as references, keeping paging stable."""
    if isinstance(value, list):
        return [pack_output(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: pack_output(item) for key, item in value.items() if key != '_source_ref'}
    if '_source_ref' in value and 'content' in value:
        result['content'] = {'_frozen_text': value['_source_ref']}
    return result


def unpack_output(value: Any, reader: SourceReader) -> Any:
    if isinstance(value, list):
        return [unpack_output(item, reader) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'_frozen_text'}:
        return reader.read(value['_frozen_text'])
    return {key: unpack_output(item, reader) for key, item in value.items()}
