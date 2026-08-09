# trackers-common

Shared trackpy-linking and per-model tracking-tuning primitives for `particle-tracking/track.py`
and `verification/benchmark.py`, extracted to stop the two from drifting against each other (see
`docs/plans/2026-08-05-001-fix-benchmark-tracking-linker-parity-plan.md`).

Installed as a local `uv` editable path dependency by `rf-detr/`, `particle-tracking/`, and
`verification/` — unlike `detectors-common`, `verification/` installs this package directly. Its
dependencies (`trackpy`, `motmetrics`, `pandas`) have no CUDA/heavy-ML sensitivity, so there's no
need for `detectors-common`'s re-exec-then-lazy-import dance; every consumer imports it at module
scope.

## Conventions

1. **This package stays scoped to trackpy/bytetrack-linking and MOT-evaluation-support primitives
   only.** Detector loading, tiling, and model-config-merge logic belongs in `detectors-common`, not
   here — see that package's own README for why it, in turn, excludes tracking-linkage logic. Keeping
   each package narrowly scoped is what keeps either one safe to install broadly without dragging in
   unrelated dependencies.
2. **Consumers re-export under their original local names and call them unqualified.**
   `particle-tracking/track.py` re-exports `link_and_filter_tracks`/`bridge_track_gaps` at module
   scope, mirroring how it already re-exports `get_rfdetr_model` from `detectors_common`. This keeps
   existing test-mocking conventions (`monkeypatch.setattr(track, "link_and_filter_tracks", ...)`)
   working unchanged.
3. **Per-model tracking-tuning values (`tracker_defaults.yaml`) are the single source of truth** for
   both `particle-tracking/tracker_configs.py`'s generated per-model configs and
   `verification/benchmark.py`'s tracking-metrics resolution. Add a new model's tuning here, not in
   either consumer.

## Dataset scale profile derivation

`dataset_profile.py` loads a per-dataset scale profile (`size_px`/`spacing_px` — see
`dataset-profiles/README.md` for the file format; deliberately duplicated from
`detectors-common`'s own loader rather than shared, to keep the two packages independent
siblings). `scale_derivation.py` derives `search_range` and trackpy's `diameter` from that
profile via the same three-tier chain as `detectors-common`'s `box_size`/`nms_distance`/
`tile_size` (explicit config value > profile-derived value > today's hardcoded default).
`memory` resolves through the same function shape for API consistency but is a deliberate
exception — it's an occlusion/blinking-duration parameter with no spatial grounding, so it
always resolves to `tracker_defaults.yaml`'s flat per-model canonical value regardless of
the profile.

## Dependencies

`trackpy`, `motmetrics`, `pandas`, `pyyaml`. All pure-Python/pandas — no CUDA sensitivity, unlike
`detectors-common`'s `torch`/`rfdetr`-adjacent callers.
