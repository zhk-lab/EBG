"""Budgeted original-material reads, persisted for restart and continuation."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .render import render, tokens
from .storage import HarnessError, Store, dumps
from .source_refs import SourceReader, pack_output, unpack_output


def child_ref(ref: str, key: str | int) -> str:
    key = str(key).replace("~", "~0").replace("/", "~1")
    return ref + ("/" if "#" in ref else "#/") + key


class OutputPages:
    def __init__(self, store: Store, budget: int) -> None:
        self.store, self.budget = store, budget

    def save(self, owner: str, value: Any) -> str:
        data = dumps(pack_output(value))
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            kind, _, identifier = owner.partition(':')
            if kind == 'evidence':
                if not db.execute("SELECT 1 FROM tasks WHERE id=?", (identifier,)).fetchone():
                    raise HarnessError("Task was cleared while building evidence.")
            else:
                self.store.require_session(db, identifier)
            row = db.execute("SELECT id FROM outputs WHERE owner=? AND data=?", (owner, data)).fetchone()
            if row:
                return row[0]
            identifier = "O" + uuid4().hex[:12]
            db.execute("INSERT INTO outputs VALUES (?, ?, ?)", (identifier, owner, data))
        return identifier

    def resolve(self, owner: str, ref: str) -> Any:
        identifier, _, pointer = ref.partition("#")
        with self.store.connect() as db:
            row = db.execute("SELECT data FROM outputs WHERE id=? AND owner=?", (identifier, owner)).fetchone()
        if not row:
            raise HarnessError("Unknown read_ref in this task/session; use a returned reference.")
        value = json.loads(row[0])
        try:
            if pointer:
                if not pointer.startswith("/"):
                    raise ValueError
                for key in pointer[1:].split("/"):
                    key = key.replace("~1", "/").replace("~0", "~")
                    value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise HarnessError("Invalid read_ref; copy one from the tool response.") from error
        return unpack_output(value, SourceReader(self.store))

    def read(self, owner: str, ref: str, offset: int = 0) -> str:
        if offset < 0:
            raise HarnessError("offset must be nonnegative.")
        value = self.resolve(owner, ref)
        # A material may itself contain a link; keep it instead of overwriting it.
        full = ({**value, "read_ref": ref} if isinstance(value, dict) and "read_ref" not in value
                else {"read_ref": ref, "content": value})
        if offset == 0 and self.fits(full):
            return render(full)
        if isinstance(value, str):
            return self._text(ref, value, offset)
        if isinstance(value, dict) and "evidence_groups" in value:
            return self._evidence(ref, value, offset)
        if not isinstance(value, (dict, list)):
            raise HarnessError("This value cannot be paged.")
        return self._directory(ref, value, offset)

    def fits(self, value: Any) -> bool:
        return tokens(render(value)) <= self.budget

    def _directory(self, ref: str, value: dict | list, offset: int) -> str:
        items = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        if offset >= len(items):
            raise HarnessError("offset is outside this directory.")
        result = {"read_ref": ref, "offset": offset, "total": len(items), "entries": [],
                  "note": "按 read_ref 展开原文；next 读取剩余目录。"}
        if '#' in ref:
            result['root_ref'] = ref.split('#', 1)[0]
        for index in range(offset, len(items)):
            key, content = items[index]
            entry = {"key": key, "read_ref": child_ref(ref, key)}
            result["entries"].append(entry)
            result["next"] = {"read_ref": ref, "offset": index + 1}
            if not self.fits(result):
                result["entries"].pop()
                result["next"]["offset"] = index
                break
            entry["content"] = content
            if not self.fits(result):
                del entry["content"]
                if isinstance(content, dict):
                    for field in ('component', 'node', 'role', 'root'):
                        if field in content:
                            entry[field] = content[field]
                            if not self.fits(result):
                                del entry[field]
                    if 'relations' in content:
                        entry['relations_ref'] = child_ref(entry['read_ref'], 'relations')
                        if not self.fits(result):
                            del entry['relations_ref']
                if isinstance(content, dict) and isinstance(content.get('content'), str):
                    # Keep enough identity to choose an excerpt without opening
                    # another metadata directory before reaching its raw text.
                    for field, value in (
                        ('source', content.get('source')),
                        ('content_chars', len(content['content'])),
                        ('content_ref', child_ref(entry['read_ref'], 'content')),
                    ):
                        if value is not None:
                            entry[field] = value
                            if not self.fits(result):
                                del entry[field]
        else:
            result.pop("next", None)
        if not result["entries"] or not self.fits(result):
            raise HarnessError("Return budget too small for a directory entry; increase --token-budget.")
        return render(result)

    def _text(self, ref: str, content: str, offset: int) -> str:
        if not 0 <= offset < len(content):
            raise HarnessError("offset is outside this text (Unicode character offset).")
        def page(end: int) -> dict:
            result = {"read_ref": ref, "offset": offset, "total_chars": len(content),
                      "content": content[offset:end]}
            if end < len(content):
                result["next"] = {"read_ref": ref, "offset": end}
            return result
        low, high = offset, len(content)
        while low < high:
            mid = (low + high + 1) // 2
            if self.fits(page(mid)):
                low = mid
            else:
                high = mid - 1
        if low == offset:
            raise HarnessError("Return budget too small for text; increase --token-budget.")
        return render(page(low))

    def _evidence(self, ref: str, value: dict, offset: int) -> str:
        groups = list(value["evidence_groups"].items())
        checklist = 'questions' if 'questions' in value else 'requirements'
        if offset >= len(groups):
            raise HarnessError("offset is outside the evidence directory.")
        result = {
            "read_ref": ref,
            "beg_disclose_prompt": "核对原文后披露；未展开不等于未匹配。按引用继续调用本工具。",
            "instructions_ref": child_ref(ref, "beg_disclose_prompt"),
            checklist: {"read_ref": child_ref(ref, checklist)},
            "evidence_groups": {}, "total_groups": len(groups),
        }
        if 'check_id' in value:
            result['check_id'] = value['check_id']
        if 'changes' in value:
            result['changes'] = value['changes']
        if 'repository' in value:
            result['repository'] = value['repository']
        if 'trace_context' in value:
            result['trace_context'] = {'read_ref': child_ref(ref, 'trace_context')}
        if 'research_context' in value:
            result['research_context'] = {'read_ref': child_ref(ref, 'research_context')}
        if 'linked_artifacts' in value:
            linked = value['linked_artifacts']
            result['linked_artifacts'] = {
                'note': '引用到的配置与输入来源材料；用于核实结论前提，按引用展开。',
                'read_ref': child_ref(ref, 'linked_artifacts'),
            }
            navigation = [{'path': item['path'], 'from': item['from'], 'read_ref': item['read_ref']}
                          for item in linked['items']]
            result['linked_artifacts']['files'] = navigation
            if tokens(render(result)) > self.budget // 2:
                del result['linked_artifacts']['files']
            relations = linked.get('table_relations')
            if relations:
                # Keep derived source relationships discoverable even when code
                # excerpts force pagination; preserve their evidence caveats.
                relation_ref = child_ref(child_ref(ref, 'linked_artifacts'), 'table_relations')
                result['linked_artifacts']['table_relations'] = {**relations, 'read_ref': relation_ref}
                if tokens(render(result)) > self.budget // 2:
                    result['linked_artifacts']['table_relations'] = {
                        'read_ref': relation_ref,
                        'group_count': len(relations.get('groups', [])),
                    }
        if 'history_context' in value:
            result['history_context'] = value['history_context']
        # Prefer a full checklist when it leaves space for the evidence directory.
        short = result[checklist]
        result[checklist] = value[checklist]
        if tokens(render(result)) > self.budget // 2:
            result[checklist] = short
        for index in range(offset, len(groups)):
            key, group = groups[index]
            pointer = child_ref(child_ref(ref, "evidence_groups"), key)
            result["evidence_groups"][key] = {"read_ref": pointer}
            result["next"] = {"read_ref": ref, "offset": index + 1}
            if not self.fits(result):
                del result["evidence_groups"][key]
                result["next"]["offset"] = index
                break
            result["evidence_groups"][key] = group
            if not self.fits(result):
                result["evidence_groups"][key] = {"read_ref": pointer}
        else:
            result.pop("next", None)
        if not result["evidence_groups"] or not self.fits(result):
            # A very small budget still provides a pageable directory.
            return self._directory(child_ref(ref, "evidence_groups"), value["evidence_groups"], offset)
        return render(result)
