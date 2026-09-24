# Case provenance

These local fixtures condense mechanisms described in public projects and
discussions. They are controlled examples rather than complete reproductions
of the source projects. The references below retain the design provenance
recorded with the cases.

| Case | Mechanism | Primary reference |
| --- | --- | --- |
| Ambiguity | Competing metrics need an explicit acceptance rule | [Datadog autoresearch](https://www.datadoghq.com/blog/llm-experimentation-autoresearch/) |
| Adjustment | Serial/parallel fallback can change random streams | [RocketPy issue 1053](https://github.com/RocketPy-Team/RocketPy/issues/1053) |
| Data leakage | Optimization must preserve held-out data boundaries | [Autoresearch issue 491](https://github.com/karpathy/autoresearch/issues/491) and [MMM research](https://github.com/lucianfialho/mmm-research) |
| API verification | Successful local checks need not establish live execution | [Agent coworker test audit](https://github.com/mweinbach/agent-coworker/blob/main/test-suite-audit.md) |
| Search budget | Unequal attempts can affect apparent improvements | [Autoresearch discussion 340](https://github.com/karpathy/autoresearch/discussions/340) |

Supporting design references: [AutoRAGsearch](https://github.com/Smasko7/AutoRAGsearch),
[random seed engineering](https://github.com/karpathy/autoresearch/issues/131),
[validation metric pitfalls](https://github.com/karpathy/autoresearch/issues/599),
[ds_ex](https://github.com/nshkrdotcom/ds_ex),
[OpenClaw fallback issue](https://github.com/openclaw/openclaw/issues/76492),
[Archon smoke-test issue](https://github.com/coleam00/Archon/issues/996),
[reflective autoresearch](https://github.com/Hzz-Git/reflective-autoresearch),
[confirmation design](https://github.com/karpathy/autoresearch/discussions/317),
and [FSRS optimization](https://github.com/Expertium/fsrs-rs-speed-autoresearch).
