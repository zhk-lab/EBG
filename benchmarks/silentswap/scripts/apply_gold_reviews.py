#!/usr/bin/env python3
"""Apply completed primary, secondary, and adjudicator records to gold.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import review_config as review


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
REVIEW_LOG = ROOT / "review" / "reviews.jsonl"
CONTENT_FIELDS = (
    "proposed_swap_type",
    "original_semantics",
    "swapped_semantics",
    "why_different",
    "evidence",
)
DECISIONS = {"accept", "revise", "rewrite", "reject"}


def load_records() -> dict[tuple[int, int, str], dict[str, Any]]:
    records = {}
    for line in REVIEW_LOG.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (
            record["sample_number"],
            record["swap_number"],
            record["reviewer_role"],
        )
        if key in records:
            raise ValueError(f"duplicate review record: {key}")
        records[key] = record
    return records


def require_completed(record: dict[str, Any]) -> None:
    if record["review_status"] != "completed":
        raise ValueError(
            f"sample {record['sample_number']} {record['reviewer_role']} review is incomplete"
        )
    if record["decision"] not in DECISIONS:
        raise ValueError(
            f"sample {record['sample_number']}: unsupported review decision"
        )
    if record["decision"] != "reject" and any(
        record[field] is None for field in CONTENT_FIELDS
    ):
        raise ValueError(
            f"sample {record['sample_number']}: completed review content is missing"
        )
    if (
        record["decision"] != "reject"
        and record["proposed_swap_type"] not in review.SAMPLES_BY_SWAP_TYPE
    ):
        raise ValueError(
            f"sample {record['sample_number']}: unsupported reviewed swap_type"
        )


def content(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in CONTENT_FIELDS)


def gold_content(gold: dict[str, Any]) -> tuple[Any, ...]:
    return (
        gold["swap_type"],
        gold["original_semantics"],
        gold["swapped_semantics"],
        gold["why_different"],
        gold["evidence"],
    )


def validate_accept_content(
    sample_number: int,
    swap_number: int,
    final: dict[str, Any],
    gold: dict[str, Any],
) -> None:
    if final["decision"] == "accept" and content(final) != gold_content(gold):
        raise ValueError(
            f"sample {sample_number} swap {swap_number}: "
            "accept review cannot change Gold content"
        )


def gold_source(decisions: set[str]) -> str:
    if "rewrite" in decisions:
        return "human_rewritten"
    if "revise" in decisions:
        return "model_draft_human_revised"
    return "model_draft_human_verified"


def resolve_swap(
    sample_number: int,
    swap_number: int,
    records: dict[tuple[int, int, str], dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    primary = records[(sample_number, swap_number, "primary")]
    require_completed(primary)
    final = primary
    decisions = {primary["decision"]}
    status = "single_reviewed"

    if sample_number in review.DOUBLE_REVIEW_SAMPLES:
        secondary = records[(sample_number, swap_number, "secondary")]
        require_completed(secondary)
        decisions.add(secondary["decision"])
        if primary["decision"] == secondary["decision"] and content(primary) == content(
            secondary
        ):
            status = "double_reviewed"
        else:
            key = (sample_number, swap_number, "adjudicator")
            if key not in records:
                raise ValueError(
                    f"sample {sample_number} swap {swap_number}: disagreement requires adjudication"
                )
            final = records[key]
            require_completed(final)
            decisions = {final["decision"]}
            status = "adjudicated_after_disagreement"

    if final["decision"] == "reject":
        return final, "model_generated", "rejected"
    return final, gold_source(decisions), status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reapply-reviewed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records()
    updates = []
    reviewed = 0
    for sample_number in range(1, 101):
        path = DATA_ROOT / str(sample_number) / "gold.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for swap_number, gold in enumerate(document["swaps"], 1):
            if gold["adjudication_status"] != "pending" and not args.reapply_reviewed:
                continue
            final, source, status = resolve_swap(
                sample_number, swap_number, records
            )
            if status != "rejected":
                validate_accept_content(sample_number, swap_number, final, gold)
                gold.update(
                    {
                        "swap_type": final["proposed_swap_type"],
                        "original_semantics": final["original_semantics"],
                        "swapped_semantics": final["swapped_semantics"],
                        "why_different": final["why_different"],
                        "evidence": final["evidence"],
                    }
                )
            gold["gold_source"] = source
            gold["adjudication_status"] = status
            reviewed += 1
        updates.append((path, document))

    for path, gold in updates:
        path.write_text(
            json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Applied {reviewed} completed Gold review decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
