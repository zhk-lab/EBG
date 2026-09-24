# Evidence-Grounded Behavior Graph

Construction and presentation follow deterministic rules; the monitoring model
makes the semantic judgment.

| Stage | Source | Responsibility |
| --- | --- | --- |
| Evidence extraction | `src/ebg/evidence_intake.py` | Preserve visible source content and locations |
| Behavior formation | `src/ebg/behavior_atomization.py` | Group condition/operation/result or demand/action/response evidence |
| Relations | `src/ebg/relation_linking.py` | Link supported relationships between scopes |
| Graph assembly | `src/ebg/graph_assembly.py` | Assemble and validate the graph |
| Directory ranking | `src/ebg/behavior_directory.py` | Rank repository entries using task cues |
| Repository views | `src/ebg/local_graph_retrieval.py` | Select seeds, expand neighbors, render components |
| Trajectory views | `src/tracereview/runner.py` | Present Current and History scopes with evidence |

AgentLoop implements repository `search`, `read`, and `finish`. TraceReview
performs one monitoring call over the visible trajectory. RepoGraph supplies
the repository-only structural baseline.

Gold labels, deleted requirements, construction patches, and target user feedback
are excluded from monitor input. Multilingual matching expressions and fixtures
remain where they implement language support.

The independently packaged harness has checkpointed snapshots and session state.
Its embedded EBG implementation includes application-specific behavior and is
not silently substituted for the main-experiment implementation.
