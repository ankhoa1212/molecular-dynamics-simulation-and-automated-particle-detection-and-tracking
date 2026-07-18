# detectors-common

Shared detector-loading, tiling, and config-merge primitives for RF-DETR and
LodeSTAR, extracted from `particle-tracking/track.py` and
`verification/benchmark.py` to stop the two from drifting against each other
(see `docs/plans/2026-07-17-001-refactor-consolidate-verification-particle-tracking-plan.md`).

Installed as a local `uv` editable path dependency by `rf-detr/` and
`particle-tracking/`. `verification/` never installs it directly — it only
becomes reachable after `verification/benchmark.py`'s existing cross-venv
re-exec lands in one of the other two venvs.

## Conventions

Two rules keep this package from becoming exactly the kind of duplicated,
drifted code it was created to eliminate. Follow them even if this plan
document is long gone by the time you're reading this:

1. **Consumers re-export under their original local names and call them
   unqualified — never `detectors_common.x(...)` inline.** `particle-tracking/track.py`
   re-exports at module scope (its venv has `detectors-common` installed
   natively). `verification/benchmark.py` cannot do the same — its venv never
   installs `detectors-common`, so it defines a thin wrapper per function that
   imports `detectors_common` lazily inside the wrapper body instead of at
   module scope. Both patterns exist so existing test-mocking conventions
   (`monkeypatch.setattr(track, "get_rfdetr_model", ...)`,
   `mock.patch.object(benchmark, "get_lodestar_model", ...)`) keep working
   unchanged.
2. **This package stays scoped to detector loading, tiling, and config-merge
   primitives only.** Tracking-linkage logic, MLflow helpers, or other
   pipeline-stage code belongs elsewhere, even once a shared package exists
   to make it tempting to add "just one more thing" here. Keeping this
   package free of `torch`/`rfdetr`/`deeplay` as its own dependencies (they
   stay lazy, function-local imports) is what makes it safe to install into
   any CUDA-sensitive venv — a heavier package would undermine that.

## Dependencies

`numpy` and `supervision` only. `torch`, `rfdetr`, and `deeplay` are never
declared here — every function that needs them imports lazily inside its own
body, assuming the caller's environment already has them importable (via its
own dependencies, or via this package's site-packages-injection helpers).
