# Task 8 Picked Metric Fix Report

## Outcome

The evaluator now exposes the final physical picked count required by Task 8 horizon ranking:

- `EpisodeResult.picked` serializes the per-episode count into normal evaluation JSON.
- `EvalSummary.mean_picked` provides the explicit aggregate used to compare horizons.
- `run_episode` reads the final value directly from `EventTracker`.
- Episode logging includes `picked=N` alongside placed count and other rollout details.

The metric remains read-only evaluator truth. It is not passed to `PolicyStepper`, policy observations,
actions, narration metrics, or preprocessing.

## Compatibility

Both fields have zero-valued defaults and are appended after all existing dataclass fields. Existing
keyword construction and the complete legacy positional argument order therefore remain valid.

## Strict TDD evidence

The first non-simulation run failed in five expected locations because `EpisodeResult.picked` and
`EvalSummary.mean_picked` did not exist, `run_episode` omitted the tracker count, and evaluation
could not serialize or log it. After the minimal propagation change, all non-simulation evaluator
tests passed.

A subsequent positional-compatibility regression test failed against the initial insertion order:
the old `false_pick_done=2` positional value was interpreted as `picked=2`. Moving the new defaulted
fields to the end made that regression and the full non-simulation set pass.

## Verification

- `.venv/bin/python -m pytest tests/sim/test_evaluate.py -m 'not sim' -q`
- `MUJOCO_GL=egl .venv/bin/python -m pytest tests/sim/test_evaluate.py -m sim -q`
- `.venv/bin/python -m ruff check src/lerobot_policy_snvla/sim/evaluate.py tests/sim/test_evaluate.py`
- `git diff --check`

The pre-existing untracked `outputs/` directory was not read, modified, staged, or committed by this
fix.
