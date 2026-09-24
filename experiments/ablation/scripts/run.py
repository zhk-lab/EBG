"""Prepare inputs offline, predict, or judge the three ablations; resume by sample."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv
from agentloop import RunStore
from agentloop.config import DEFAULT_CONFIG
from agentloop.protocol import FinishAction, parse_action
from agentloop.runner import AgentLoop
from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend
from agentloop.context import FastTokenCounter
from agentloop.errors import RetryableModelError
from agentloop.silentswap_source_review import (
    build_review, freeze_json, load_stage1, parse_review_prediction, run_review, save_request,
)
from ebg.behavior_directory import DIRECTORY_ENCODING
from ebg.evidence_intake import load_visible_bundle
from evaluation_core.contracts import PredictionError, RepoPredictionContract, load_prediction_schema
from scripts.main import judge, predict
from scripts.main.prepare import token_counter
from tracereview import render_trace_view
from tracereview.runner import validate_trace_prediction, _unique_object, _reject_constant

from ablate_behavior import EvidenceBackend, render_trace as evidence_trace
from ablate_graph import BehaviorListBackend, render_trace as behavior_trace
from ablate_task import FixedEntryBackend

VARIANTS = ("full", "no_behavior", "no_graph", "no_task")
BENCHMARKS = ("specgap", "silentswap", "feedbacktrace")
BACKENDS = {"full": GraphBackend, "no_behavior": EvidenceBackend,
            "no_graph": BehaviorListBackend, "no_task": FixedEntryBackend}
TRACE_RENDERERS = {"full": render_trace_view, "no_behavior": evidence_trace, "no_graph": behavior_trace}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    RunStore(Path(path).parent)._write_json(Path(path), value)


def supported(benchmark, variant):
    return benchmark != "feedbacktrace" or variant != "no_task"


def load_repo(benchmark, input_id, variant):
    artifacts = ROOT / "data" / "prepared" / benchmark / "artifacts"
    bundle = load_visible_bundle(artifacts / "visible_bundles" / input_id)
    graph = read_json(artifacts / "behavior_graphs" / input_id / "behavior_graph.json")
    directory = read_json(artifacts / "behavior_directories" / input_id / "ranked_directory.json")
    backend = BACKENDS[variant](bundle, graph, directory, count_tokens=token_counter(DIRECTORY_ENCODING))
    return bundle, graph, backend


def prompt_text(benchmark, variant):
    return (HERE / "prompts" / f"{benchmark}_{variant}.txt").read_text(encoding="utf-8")


def preview(benchmark, variant, input_id, output):
    """Save actual read output as well as the initial request; never create an API client."""
    constructed = []

    def factory(bundle, graph, directory, **kwargs):
        backend = BACKENDS[variant](bundle, graph, directory, **kwargs)
        constructed.append((backend, bundle, graph, directory, kwargs))
        return backend

    args = argparse.Namespace(benchmark=benchmark, input_id=input_id, arm="graph",
                              artifact_root=None, output=output, schema_root=ROOT / "schemas",
                              prepare_only=True)
    result = predict._run(args, backend_factory=factory,
                          trace_renderer=TRACE_RENDERERS.get(variant),
                          task_prompt=prompt_text(benchmark, variant))
    if benchmark == "feedbacktrace":
        graph = read_json(ROOT / "data" / "prepared" / benchmark / "artifacts/behavior_graphs" /
                          input_id / "behavior_graph.json")
        full, changed = render_trace_view(graph), TRACE_RENDERERS[variant](graph)
        (output / "view.txt").write_text(changed.text, encoding="utf-8")
        result.update(input_changed=full.text != changed.text, evidence_ids_preserved=full.evidence_ids == changed.evidence_ids)
    else:
        backend, bundle, graph, original_directory, options = constructed[0]
        full = backend if variant == "full" else GraphBackend(bundle, graph, original_directory, **options)
        directory = getattr(backend, "effective_directory", backend._retriever.directory)
        ids = [e["read_id"] for e in directory["entries"]
               if e["read_id"] in backend.initial_ids][:DEFAULT_CONFIG.max_read_ids]
        from agentloop.context import FastTokenCounter
        kwargs = dict(token_budget=DEFAULT_CONFIG.tool_result_budget,
                      max_atomic_unit_tokens=DEFAULT_CONFIG.max_atomic_unit_tokens,
                      count_tokens=FastTokenCounter(DEFAULT_CONFIG.token_estimator).count_text)
        shown = backend.read(ids, **kwargs)
        full_shown = full.read(ids, **kwargs)
        rendered = render_tool_result(shown)
        (output / "read.txt").write_text(rendered, encoding="utf-8")
        write_json(output / "read.json", shown.to_dict())
        result.update(read_ids=ids, input_changed=(backend.initial_index != full.initial_index or
                                                   rendered != render_tool_result(full_shown)))
    if variant != "full" and not result["input_changed"]:
        result.update(status="no_contrast", failure="selected preview inputs are identical to full EBG")
    write_json(output / "preview.json", result)
    return result


def stage_root(results, model, benchmark, variant, stage="stage1"):
    base = Path(results) / model / benchmark / variant
    return base / stage if benchmark == "silentswap" else base


def make_manifest(path, benchmark, variant, ids, profile, prompt):
    value = {"experiment_name": path.name, "phase": "full", "split_id": "ablation_main_key",
             "benchmarks": [benchmark], "arms": ["graph"], "selected_ids": {benchmark: ids},
             "prepare_only": False, "variant": variant, "implementation_version": 1,
             "task_prompt": prompt, "runner_limits": DEFAULT_CONFIG.public_dict(), **profile}
    freeze_json(path / "manifest.json", value)
    return value


def predict_sample(config, benchmark, variant, model, input_id, output):
    profile = config["models"][model][benchmark]
    batch_path = output / "runs" / input_id / "batch_result.json"
    if batch_path.exists():
        previous = read_json(batch_path)
        if previous["status"] == "complete" and (batch_path.parent / "prediction.json").exists():
            return {**previous, "resumed": True}
    batch = predict.BatchConfig(experiment_name=output.name, experiment_root=output.parent,
                                split_file=None, phase="full", benchmarks=(benchmark,), arms=("graph",),
                                model=profile["model"], base_url=profile["base_url"],
                                api_key_env=profile["api_key_env"], request_options=profile["request_options"],
                                timeout=config["timeout"])

    def run_one(args):
        return predict._run(args, backend_factory=BACKENDS[variant],
                            trace_renderer=TRACE_RENDERERS.get(variant),
                            task_prompt=prompt_text(benchmark, variant))

    result = predict._run_job(batch, (benchmark, "graph", input_id), run_one, lambda _: None)
    if result["status"] == "failed" and benchmark in predict.RETRY_BENCHMARKS and not result.get("auto_retried"):
        result = predict._run_job(batch, (benchmark, "graph", input_id), run_one, lambda _: None, attempt=2)
    if result.get("failure") in {"invalid_finish_grounding_or_content", "finish_format_retries_exhausted"} and benchmark in predict.RETRY_BENCHMARKS:
        result = correct_repo_sample(config, model, benchmark, input_id, batch_path.parent, result)
    if (result["status"] == "failed" and benchmark in predict.RETRY_BENCHMARKS
            and result.get("auto_retried")
            and any(term in str(result.get("failure")) for term in
                    ("PredictionGroundingError", "invalid_finish_grounding", "finish_format"))):
        result = retry_repo_with_reads(batch, benchmark, input_id, run_one, batch_path.parent, result)
    if result["status"] == "failed" and benchmark == "feedbacktrace":
        state_path = batch_path.parent / "state.json"
        if state_path.exists() and read_json(state_path).get("failure") == "invalid_prediction":
            result = correct_trace_sample(config, model, variant, input_id, batch_path.parent, result)
    return result


def retry_repo_with_reads(batch, benchmark, input_id, run_one, root, previous):
    """One extra full attempt can read missing evidence; preserve all previous attempts."""
    previous_path = root / "attempt_3/original_batch_result.json"
    if not previous_path.exists():
        write_json(previous_path, previous)
    result = predict._run_job(batch, (benchmark, "graph", input_id), run_one,
                              lambda _: None, attempt=3)
    usages = [predict.collect_response_usage(p)
              for attempt in root.glob("attempt_*") if attempt.is_dir()
              for p in [attempt, *attempt.glob("grounding_correction_*")]]
    result.update(auto_retried=True, additional_read_attempt=3,
                  actual_usage={key: sum(u[key] for u in usages) for key in usages[0]},
                  initial_failure=previous.get("initial_failure"),
                  initial_format_failure=previous.get("initial_format_failure"))
    if result["status"] == "complete":
        result["repair_source_attempt"] = 3
    write_json(root / "batch_result.json", result)
    return result


def correction_completion(store, messages, client, *, max_output_tokens):
    """Save a bounded correction call and replay its response after interruption."""
    count = FastTokenCounter(DEFAULT_CONFIG.token_estimator).count_messages(messages)
    saved = store.load_response(1)
    if saved is not None:
        request = store.load_request(1, saved["provider_retry"])
        if request is None or request["messages"] != messages:
            raise ValueError("correction response does not match its saved request")
        return saved["content"]
    for retry in range(DEFAULT_CONFIG.network_retries + 1):
        store.save_request_if_unchanged(1, messages=messages, token_count=count,
                                        compression={"mode": "validation_correction"}, provider_retry=retry)
        if (store.root / "provider_failures" / f"turn_001_retry_{retry}.json").exists():
            continue
        try:
            completion = client.complete(messages, max_output_tokens=max_output_tokens)
        except RetryableModelError as error:
            store.save_provider_failure(1, provider_retry=retry, error=str(error),
                                        raw_response=error.raw_response, usage=error.usage)
            continue
        store.save_response(1, content=completion.content, raw_response=completion.raw_response,
                            usage=completion.usage, provider_retry=retry)
        return completion.content
    raise RuntimeError("correction network retries exhausted")


def correct_repo_sample(config, model, benchmark, input_id, root, previous):
    """Correct a finish answer against the same reading history, without hidden Gold."""
    attempts = sorted(root.glob("attempt_*"), key=lambda p: int(p.name.split("_")[-1]))
    attempt = attempts[-1]
    state = read_json(attempt / "state.json")
    terminal = state["terminal_record"]
    store = RunStore(attempt)
    request = store.load_request(terminal["turn"], terminal["provider_retry"],
                                 format_retry=terminal.get("format_retry", 0))
    messages, content = request["messages"], terminal["raw_response"]
    bundle = load_visible_bundle(ROOT / "data" / "prepared" / benchmark / "artifacts/visible_bundles" / input_id)
    contract = RepoPredictionContract(benchmark, state["finish_contract"]["prediction_schema"],
                                      source_texts={a.path: a.content for a in bundle.repo_artifacts})
    spans = AgentLoop._observed_spans(state["records"])
    profile = config["models"][model][benchmark]
    client = predict._client(argparse.Namespace(**profile, timeout=config["timeout"]))
    prediction, attempted = None, 0
    for number in range(5):
        try:
            action = parse_action(content, max_read_ids=state["config"]["max_read_ids"],
                                  max_search_characters=state["config"]["max_search_characters"])
            if not isinstance(action, FinishAction):
                raise ValueError("return a finish action containing the complete prediction")
            prediction = contract.validate(action.prediction, input_id=input_id, observed_spans=spans)
            break
        except ValueError as error:
            failure = f"{type(error).__name__}: {error}"
        if number == 4:
            break
        attempted = number + 1
        correction = RunStore(attempt / f"grounding_correction_{attempted}")
        detail = ""
        if number >= 2:
            visible = sorted({(s.path, s.start, s.end) for s in spans})
            detail = ("\nThe successful reads expose these source intervals (path, start, end): "
                      + json.dumps(visible, ensure_ascii=False)
                      + ". Non-contiguous source blocks require separate line_ranges entries. "
                      "Do not bridge gaps containing code that was not displayed.")
        messages = [*messages, {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        f"Your finish answer failed validation: {failure}\n"
                        "Return a corrected finish action with the complete prediction. Cite only source "
                        "locations shown in the existing successful reads. Do not invent evidence, "
                        "add unsupported findings, or request another tool call." + detail
                    )}]
        if not (correction.root / "original_batch_result.json").exists():
            write_json(correction.root / "original_batch_result.json", previous)
        try:
            content = correction_completion(correction, messages, client,
                                            max_output_tokens=state["config"]["max_output_tokens"])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            break
    usage_by_attempt = []
    for source in attempts:
        usages = [predict.collect_response_usage(p) for p in [source, *source.glob("grounding_correction*")]]
        usage_by_attempt.append({key: sum(u[key] for u in usages) for key in usages[0]})
    actual = {key: sum(u[key] for u in usage_by_attempt) for key in usage_by_attempt[0]}
    usage = usage_by_attempt[-1] if prediction is not None else {key: 0 for key in actual}
    result = {**previous, "status": "complete" if prediction is not None else "failed",
              "failure": None if prediction is not None else failure,
              "usage": usage, "actual_usage": actual, "validation_corrections": attempted,
              "usage_scope": "accepted_attempt_with_grounding_correction"}
    if prediction is not None:
        # Stage two must use the reading history of the corrected answer.
        result["repair_source_attempt"] = int(attempt.name.split("_")[-1])
        write_json(root / "prediction.json", prediction)
    write_json(root / "batch_result.json", result)
    return result


def correct_trace_sample(config, model, variant, input_id, root, previous):
    """Ask for valid Evidence IDs without adding hidden evidence or changing the task."""
    state = read_json(root / "state.json")
    graph = read_json(ROOT / "data/prepared/feedbacktrace/artifacts/behavior_graphs" / input_id / "behavior_graph.json")
    ids = set(TRACE_RENDERERS[variant](graph).evidence_ids)
    terminal = state["terminal_record"]
    messages = RunStore(root).load_request(1, terminal["provider_retry"])["messages"]
    content = terminal["raw_response"]
    profile = config["models"][model]["feedbacktrace"]
    client = predict._client(argparse.Namespace(**profile, timeout=config["timeout"]))
    prediction = None
    attempted = 0
    for number in range(3):
        try:
            body = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(body, dict):
                raise ValueError("response must be a JSON object")
            if "verdict" in body:
                raise ValueError("verdict is runner-owned; omit it from the response")
            prediction = validate_trace_prediction(
                {"verdict": "KEY", **body}, input_id=input_id, schema=state["prediction_schema"],
                selectable_evidence_ids=ids)
            break
        except ValueError as error:
            failure = f"{type(error).__name__}: {error}"
        if number == 2:
            break
        attempted = number + 1
        correction = RunStore(root / f"format_correction_{number + 1}")
        messages = [*messages, {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        f"Your answer failed validation: {failure}\n"
                        "Return the complete answer using the original schema. Copy Evidence IDs exactly "
                        "from the supplied input; do not invent IDs or use hidden evidence. "
                        "The response must contain only verification_point (a string), "
                        "supporting_evidence_ids (an array of one or two evidence ID strings), "
                        "and criticality (must_disclose or worth_disclose). "
                        "Do not return a trace summary or include input_id, benchmark, or verdict. "
                        "Answer the original decision-review task using the supplied trace."
                    )}]
        if not (correction.root / "original_batch_result.json").exists():
            write_json(correction.root / "original_batch_result.json", previous)
        try:
            content = correction_completion(correction, messages, client,
                                            max_output_tokens=state["config"]["max_output_tokens"])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            break
    usages = [predict.collect_response_usage(p) for p in [root, *sorted(root.glob("format_correction_*"))]]
    usage = {key: sum(u[key] for u in usages) for key in usages[0]}
    result = {**previous, "status": "complete" if prediction is not None else "failed",
              "failure": None if prediction is not None else failure, "usage": usage,
              "actual_usage": usage, "validation_corrections": attempted}
    if prediction is not None:
        write_json(root / "prediction.json", prediction)
    write_json(root / "batch_result.json", result)
    return result


def repair_source_review_format(review, store):
    """Remove one redundant closing delimiter at the JSON tail, then fully validate."""
    responses = sorted((store.root / "responses").glob("turn_001*.json"))
    if not responses:
        return False
    content = read_json(responses[-1])["content"]
    try:
        json.loads(content)
        return False
    except json.JSONDecodeError as error:
        position = error.pos
    # Never repair strings, values, separators, missing delimiters or truncated output.
    if (position >= len(content) or content[position] not in "}]"
            or any(c not in "}] \t\r\n" for c in content[position:])):
        return False
    corrected = content[:position] + content[position + 1:]
    contract = RepoPredictionContract("silentswap", review.schema, source_texts=review.sources)
    try:
        prediction = contract.validate(parse_review_prediction(corrected),
                                       input_id=review.input_id, observed_spans=review.spans)
    except ValueError:
        return False
    write_json(store.root / "format_repair.json", {
        "source_response": str(responses[-1].relative_to(store.root)),
        "removed_character": content[position], "position": position,
        "corrected_content": corrected,
    })
    write_json(store.root / "prediction.json", prediction)
    store.save_state({"input_id": review.input_id, "status": "complete",
                      "format_repair": "single_redundant_trailing_delimiter"})
    return True


def correct_source_review(review, client, store, error):
    """Retry invalid stage-two output with the same input and strict contract."""
    responses = sorted((store.root / "responses").glob("turn_001*.json"))
    previous_response = read_json(responses[-1])
    content = previous_response["content"]
    output_limit = DEFAULT_CONFIG.max_output_tokens
    contract = RepoPredictionContract("silentswap", review.schema, source_texts=review.sources)
    messages = review.messages
    failure = str(error)
    for number in (1, 2, 3, 4, 5):
        response_content = content
        if number == 5:
            choices = previous_response.get("raw_response", {}).get("choices", [])
            draft = next((choice.get("message", {}).get("reasoning_content")
                          for choice in choices if choice.get("finish_reason") == "length"), None)
            if content or not draft:
                break
            messages = review.messages
            response_content = "Unfinished model draft, not additional source evidence:\n" + draft
        if any(choice.get("finish_reason") == "length"
               for choice in previous_response.get("raw_response", {}).get("choices", [])):
            output_limit = DEFAULT_CONFIG.max_output_tokens * 2
        correction = RunStore(store.root / f"validation_correction_{number}")
        detail = ""
        if number > 2:
            try:
                json.loads(content)
            except json.JSONDecodeError as parse_error:
                detail = (f"\nJSON parser stopped at character {parse_error.pos}. "
                          f"Nearby text: {content[max(0, parse_error.pos - 80):parse_error.pos + 80]!r}. "
                          "Check matching braces, brackets, commas and string escaping. "
                          "Fix the JSON syntax without changing the answer's factual content.")
            if not content and any(choice.get("finish_reason") == "length"
                                   for choice in previous_response.get("raw_response", {}).get("choices", [])):
                detail += ("\nThe preceding completion exhausted its output budget without returning an answer. "
                           "Keep deliberation brief and reserve enough output budget for the final JSON. "
                           "Complete the original task and return the required prediction object in this response.")
        messages = [*messages, {"role": "assistant", "content": response_content},
                    {"role": "user", "content": (
                        f"Your prediction failed validation: {failure}\n"
                        "Return the complete prediction JSON using the original schema and only "
                        "the supplied document and source. Do not add unsupported locations or claims."
                        + detail
                    )}]
        content = correction_completion(correction, messages, client,
                                        max_output_tokens=output_limit)
        previous_response = correction.load_response(1) or {}
        try:
            prediction = contract.validate(parse_review_prediction(content),
                                           input_id=review.input_id, observed_spans=review.spans)
        except ValueError as error:
            failure = str(error)
            write_json(correction.root / "validation_error.json", {"error": failure})
            continue
        write_json(store.root / "prediction.json", prediction)
        store.save_state({"input_id": review.input_id, "status": "complete",
                          "validation_corrections": number})
        return prediction
    raise ValueError(f"source-review correction failed validation: {failure}")


def source_review_sample(config, model, variant, input_id, stage1, stage2):
    first = stage1 / "runs" / input_id
    if not (first / "prediction.json").exists():
        return {"benchmark": "silentswap", "input_id": input_id, "status": "blocked",
                "failure": "stage1 prediction is not available"}
    store = RunStore(stage2 / "runs" / input_id)
    result_path = store.root / "batch_result.json"
    if result_path.exists() and (store.root / "prediction.json").exists():
        saved = read_json(result_path)
        if saved["status"] == "complete":
            return {**saved, "resumed": True}
    try:
        prediction, state, _ = load_stage1(stage1 / "runs" / input_id)
        bundle, _, backend = load_repo("silentswap", input_id, variant)
        directory = getattr(backend, "effective_directory", backend._retriever.directory)
        review = build_review(bundle, prediction, directory, state=state,
                              prompt=(HERE / "prompts/silentswap_stage2.txt").read_text(encoding="utf-8"))
        write_json(store.root / "selection.json", review.selection)
        save_request(store, review.messages)
        profile = config["models"][model]["silentswap"]
        client = predict._client(argparse.Namespace(**profile, timeout=config["timeout"]))
        try:
            run_review(review, client, store)
        except ValueError as error:
            if "source-review prediction is invalid:" not in str(error):
                raise
            if not repair_source_review_format(review, store):
                correct_source_review(review, client, store, error)
        result = {"benchmark": "silentswap", "input_id": input_id, "status": "complete"}
    except Exception as error:
        result = {"benchmark": "silentswap", "input_id": input_id, "status": "failed",
                  "failure": f"{type(error).__name__}: {error}"}
    usages = [predict.collect_response_usage(p)
              for p in [store.root, *sorted(store.root.glob("validation_correction_*"))]]
    result["usage"] = {key: sum(u[key] for u in usages) for key in usages[0]}
    result["actual_usage"] = dict(result["usage"])
    write_json(result_path, result)
    return result


def judge_sample(batch, benchmark, input_id, module, factory):
    result = judge._run_job(batch, benchmark, "graph", input_id, module, factory)
    if batch.deferred_retries and judge._needs_deferred_retry(batch, result):
        result = judge._run_deferred_retry(batch, result, module, factory)
    if result["status"] == "failed" and "JudgeNetworkRetriesExhausted" in str(result.get("failure")):
        # Extend saved network attempts without deleting failures or replaying paid responses.
        for extra in (1, 2, 3):
            if extra > 1:
                time.sleep(30 * (extra - 1))
            recovery = replace(batch, network_retries=batch.network_retries + extra)
            result = judge._run_job(recovery, benchmark, "graph", input_id, module, factory)
            if result["status"] != "failed" or "JudgeNetworkRetriesExhausted" not in str(result.get("failure")):
                break
    if result["status"] == "failed" and "JudgeContentValidationError" in str(result.get("failure")):
        result = correct_judge_sample(batch, benchmark, input_id, module, factory, result)
    return result


def correct_judge_sample(batch, benchmark, input_id, module, factory, previous):
    """Return validation feedback to the judge, keeping predictions and scoring rules fixed."""
    root = batch.judge_root / input_id
    original = read_json(root / "judge_input.json")
    gold = judge._normalize_gold(benchmark, read_json(original["gold_path"]))
    prediction = original["prediction"]
    attempts = sorted([*root.glob("attempts/*.json"), *root.glob("retry_1/attempts/*.json")])
    content = next((read_json(p).get("content") for p in reversed(attempts)
                    if read_json(p).get("content")), None)
    messages = original["messages"]
    failure = previous.get("failure", "Judge output violates the scoring contract")
    for number in (1, 2):
        correction = root / f"alignment_correction_{number}"
        saved_input = correction / "judge_input.json"
        if saved_input.exists():
            messages = read_json(saved_input)["messages"]
        else:
            messages = [*messages]
            if content:
                messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                f"Your judgment failed validation: {failure}\n"
                "Return a corrected complete JSON judgment following the original scoring rules. "
                "Use only allowed IDs and score values. Candidate alignments must be one-to-one. "
                "Do not change the candidate prediction, gold data, or scoring criteria."
            )})
            write_json(correction / "original_status.json", previous)
            write_json(saved_input, {**original, "messages": messages})
        attempt_path = None
        try:
            content, attempt_path, resumed = judge._call_judge_stage(
                batch, stage="correction", messages=messages, sample_root=correction,
                client_factory=factory)
            result = judge._score_response(module, benchmark, gold, prediction,
                                           judge._json_object(content),
                                           judge._judge_kwargs(batch, benchmark, input_id))
            judge._validate_saved_result(module, benchmark, gold, prediction, result)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            if attempt_path is not None:
                judge._mark_attempt(attempt_path, status="invalid", failure_class="content", error=error)
            write_json(correction / "status.json", {"status": "failed", "failure": failure})
            continue
        judge._mark_attempt(attempt_path, status="valid")
        write_json(correction / "result.json", result)
        write_json(root / "result.json", result)
        record = judge._sample_record(benchmark, "graph", input_id, result=result, gold=gold,
                                      prediction=prediction, sample_root=root, resumed=resumed)
        record["validation_correction"] = number
        write_json(correction / "status.json", record)
        write_json(root / "status.json", record)
        return record
    record = {**previous, "failure": failure, "usage": judge._attempt_usage(root)}
    write_json(root / "status.json", record)
    return record


def evaluation_sample(config, model, benchmark, variant, input_id, roots, judges):
    """Run one sample's prediction and scoring stages, resuming each saved stage."""
    judge_model = config["judges"][benchmark]["model"]
    complete = True
    for root in roots:
        prediction = root / "runs" / input_id / "batch_result.json"
        judgment = root / "judges" / judge_model / input_id / "status.json"
        complete &= (prediction.exists() and judgment.exists()
                     and (prediction.parent / "prediction.json").exists()
                     and (judgment.parent / "result.json").exists()
                     and read_json(prediction)["status"] == "complete"
                     and read_json(judgment)["status"] == "complete")
    if complete:
        return {"status": "complete", "resumed": True}
    first = predict_sample(config, benchmark, variant, model, input_id, roots[0])
    stages = [{"stage": "stage1", "prediction": first["status"], "failure": first.get("failure")}]
    if first["status"] != "complete":
        return {"status": "failed", "stages": stages}
    if benchmark == "silentswap":
        second = source_review_sample(config, model, variant, input_id, *roots)
        stages.append({"stage": "stage2", "prediction": second["status"], "failure": second.get("failure")})
    for stage, (batch, module, factory) in zip(stages, judges):
        if stage["prediction"] == "complete":
            result = judge_sample(batch, benchmark, input_id, module, factory)
            stage.update(judge=result["status"], judge_failure=result.get("failure"))
    success = all(s["prediction"] == "complete" and s.get("judge") == "complete" for s in stages)
    return {"status": "complete" if success else "failed", "stages": stages}


def evaluation_progress_path(results, models, benchmarks, variants):
    """Concurrent model batches must not share a progress file or its temporary file."""
    label = "_".join(["-".join(models), "-".join(benchmarks), "-".join(variants)])
    return results / f"evaluation_progress_{label}.json"


def evaluate(config, samples, args, models, variants):
    """Share one worker pool across experiments; each sample retains its own files."""
    groups = []
    for model in models:
        for benchmark in args.benchmark or BENCHMARKS:
            ids = args.ids or samples[benchmark]
            if len(ids) != len(set(ids)) or any(i not in samples[benchmark] for i in ids):
                raise ValueError(f"IDs must be unique members of {benchmark}")
            if args.limit:
                ids = ids[:args.limit]
            for variant in variants:
                if not supported(benchmark, variant):
                    continue
                roots, judges = [], []
                for stage in (("stage1", "stage2") if benchmark == "silentswap" else ("stage1",)):
                    root = stage_root(args.results, model, benchmark, variant, stage)
                    prompt = ((HERE / "prompts/silentswap_stage2.txt").read_text(encoding="utf-8")
                              if stage == "stage2" else prompt_text(benchmark, variant))
                    make_manifest(root, benchmark, variant, samples[benchmark], config["models"][model][benchmark], prompt)
                    profile = config["judges"][benchmark]
                    batch = judge.JudgeBatchConfig(
                        experiment_name=root.name, experiment_root=root.parent, phase="full",
                        benchmarks=(benchmark,), arms=("graph",), judge_model=profile["model"],
                        base_url=profile["base_url"], api_key_env=profile["api_key_env"],
                        request_options=profile["request_options"], workers=args.workers, timeout=config["timeout"])
                    freeze_json(root / "judges" / batch.judge_model / "profile.json", profile)
                    roots.append(root)
                    judges.append((batch, judge._load_judge_module(benchmark), judge._default_client_factory(batch)))
                groups.append((model, benchmark, variant, ids, roots, judges))
    progress = {"workers": args.workers, "expected": sum(len(g[3]) for g in groups),
                "finished": 0, "successful": 0, "failed": [], "status": "running"}
    path = evaluation_progress_path(args.results, models, args.benchmark or BENCHMARKS, variants)
    write_json(path, progress)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index in range(max((len(g[3]) for g in groups), default=0)):
            for model, benchmark, variant, ids, roots, judges in groups:
                if index >= len(ids):
                    continue
                ident = {"model": model, "benchmark": benchmark, "variant": variant, "input_id": ids[index]}
                future = pool.submit(evaluation_sample, config, model, benchmark, variant, ids[index], roots, judges)
                futures[future] = ident
        for future in as_completed(futures):
            try:
                result = {**futures[future], **future.result()}
            except Exception as error:
                result = {**futures[future], "status": "failed", "failure": f"{type(error).__name__}: {error}"}
            progress["finished"] += 1
            if result["status"] == "complete":
                progress["successful"] += 1
            else:
                progress["failed"].append(result)
            write_json(path, progress)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    progress["status"] = "finished_with_failures" if progress["failed"] else "complete"
    write_json(path, progress)
    return int(bool(progress["failed"]))


def revalidate_sample(output, benchmark, input_id):
    """Recover saved failed answers under the shared grounding rule, without API calls."""
    case = output / "runs" / input_id
    batch_path = case / "batch_result.json"
    if not batch_path.exists():
        return {"input_id": input_id, "status": "pending"}
    previous = read_json(batch_path)
    if previous["status"] == "complete":
        return {"input_id": input_id, "status": "complete", "resumed": True}
    if benchmark in predict.RETRY_BENCHMARKS and not previous.get("auto_retried"):
        return {"input_id": input_id, "status": "pending_retry"}
    bundle = load_visible_bundle(ROOT / "data" / "prepared" / benchmark / "artifacts/visible_bundles" / input_id)
    contract = RepoPredictionContract(benchmark, load_prediction_schema(ROOT / "schemas", benchmark),
                                      source_texts={a.path: a.content for a in bundle.repo_artifacts})
    checked = []
    for attempt in sorted(case.glob("attempt_*")):
        state_path = attempt / "state.json"
        if not state_path.exists():
            continue
        state = read_json(state_path)
        terminal = state.get("terminal_record") or {}
        if terminal.get("failure") != "invalid_finish_grounding_or_content":
            checked.append({"attempt": attempt.name, "error": state.get("failure")})
            continue
        action = parse_action(terminal["raw_response"], max_read_ids=state["config"]["max_read_ids"],
                              max_search_characters=state["config"]["max_search_characters"])
        if not isinstance(action, FinishAction):
            continue
        try:
            prediction = contract.validate(action.prediction, input_id=input_id,
                                           observed_spans=AgentLoop._observed_spans(state["records"]))
        except PredictionError as error:
            checked.append({"attempt": attempt.name, "error": str(error)})
            continue
        # Choose the earliest valid saved answer, independently of its judge score.
        write_json(case / "validation_recheck.json", {
            "policy": "verified_non_code_gaps", "original_batch_result": previous,
            "accepted_attempt": attempt.name, "rejected_attempts": checked,
        })
        write_json(case / "prediction.json", prediction)
        result = {**previous, "status": "complete", "failure": None, "turns": state["turns"],
                  "usage": predict.collect_response_usage(attempt), "validation_rechecked": True,
                  "accepted_attempt": attempt.name}
        write_json(batch_path, result)
        return result
    write_json(case / "validation_recheck.json", {
        "policy": "verified_non_code_gaps", "original_batch_result": previous,
        "accepted_attempt": None, "rejected_attempts": checked,
    })
    return {"input_id": input_id, "status": "failed", "attempts": checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "predict", "judge", "revalidate", "evaluate"))
    parser.add_argument("--model", nargs="+", choices=("kimi-k3", "luna", "glm-5-3"))
    parser.add_argument("--benchmark", nargs="+", choices=BENCHMARKS)
    parser.add_argument("--variant", nargs="+", choices=VARIANTS)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", type=Path, default=HERE / "configs/config.json")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/analysis/ablation/results")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    config = read_json(args.config)
    samples = read_json(HERE / "configs/samples.json")
    if args.command != "prepare":
        load_dotenv(ROOT / ".env", override=False)
    models = args.model or list(config["models"])
    variants = args.variant or list(VARIANTS)
    if args.command == "evaluate":
        return evaluate(config, samples, args, models, variants)
    failures = 0
    for benchmark in args.benchmark or BENCHMARKS:
        if args.command == "revalidate" and benchmark != "specgap":
            parser.error("saved-answer revalidation currently supports --benchmark specgap")
        all_ids = samples[benchmark]
        ids = args.ids or all_ids
        if len(ids) != len(set(ids)) or any(i not in all_ids for i in ids):
            parser.error(f"IDs must be unique members of {benchmark}")
        if args.limit:
            ids = ids[:args.limit]
        for variant in variants:
            if not supported(benchmark, variant):
                print(json.dumps({"benchmark": benchmark, "variant": variant, "status": "not_applicable"}))
                continue
            if args.command == "prepare":
                for input_id in ids:
                    out = args.results / "previews" / benchmark / variant / input_id
                    try:
                        result = preview(benchmark, variant, input_id, out)
                    except Exception as error:
                        result = {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
                        write_json(out / "preview.json", result)
                    print(json.dumps({"benchmark": benchmark, "variant": variant, "input_id": input_id,
                                      **result}, ensure_ascii=False), flush=True)
                    failures += result["status"] != "prepared"
                continue
            for model in models:
                output = stage_root(args.results, model, benchmark, variant)
                if args.command == "revalidate":
                    for input_id in ids:
                        result = revalidate_sample(output, benchmark, input_id)
                        print(json.dumps({"model": model, "variant": variant, **result}), flush=True)
                        failures += result["status"] == "failed"
                    continue
                profile = config["models"][model][benchmark]
                if args.command == "predict":
                    make_manifest(output, benchmark, variant, all_ids, profile, prompt_text(benchmark, variant))
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        jobs = {pool.submit(predict_sample, config, benchmark, variant, model, i, output): i for i in ids}
                        for future in as_completed(jobs):
                            result = future.result()
                            print(json.dumps({"model": model, "variant": variant, **result}, ensure_ascii=False), flush=True)
                            failures += result["status"] != "complete"
                    if benchmark == "silentswap":
                        second = stage_root(args.results, model, benchmark, variant, "stage2")
                        make_manifest(second, benchmark, variant, all_ids, profile,
                                      (HERE / "prompts/silentswap_stage2.txt").read_text(encoding="utf-8"))
                        with ThreadPoolExecutor(max_workers=args.workers) as pool:
                            jobs = [pool.submit(source_review_sample, config, model, variant,
                                                input_id, output, second) for input_id in ids]
                            for future in as_completed(jobs):
                                result = future.result()
                                print(json.dumps({"model": model, "variant": variant, "stage": "stage2", **result}), flush=True)
                                failures += result["status"] != "complete"
                else:
                    # Judge only saved predictions; do not request absent samples or silently score them as zero.
                    stages = [output]
                    if benchmark == "silentswap":
                        stages.append(stage_root(args.results, model, benchmark, variant, "stage2"))
                    for stage in stages:
                        missing = [i for i in ids if not (stage / "runs" / i / "prediction.json").exists()]
                        if missing:
                            print(json.dumps({"status": "incomplete", "stage": str(stage), "missing": missing}))
                            failures += 1
                        available = [i for i in ids if i not in missing]
                        if not available:
                            continue
                        # The public judge batch covers its frozen sample set. For a subset use the same
                        # sample judge and retry policy, without changing the full experiment manifest.
                        profile_j = config["judges"][benchmark]
                        batch = judge.JudgeBatchConfig(
                            experiment_name=stage.name, experiment_root=stage.parent, phase="full",
                            benchmarks=(benchmark,), arms=("graph",), judge_model=profile_j["model"],
                            base_url=profile_j["base_url"], api_key_env=profile_j["api_key_env"],
                            request_options=profile_j["request_options"], workers=args.workers, timeout=config["timeout"])
                        freeze_json(stage / "judges" / batch.judge_model / "profile.json", profile_j)
                        module = judge._load_judge_module(benchmark)
                        factory = judge._default_client_factory(batch)
                        with ThreadPoolExecutor(max_workers=args.workers) as pool:
                            jobs = [pool.submit(judge_sample, batch, benchmark, i, module, factory) for i in available]
                            for future in as_completed(jobs):
                                result = future.result()
                                print(json.dumps({"model": model, "variant": variant, **result}), flush=True)
                                failures += result["status"] != "complete"
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
