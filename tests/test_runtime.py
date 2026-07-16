import json

from lerobot_policy_snvla.runtime import (
    SNVLAOutput,
    SNVLARuntime,
    SNVLARuntimeMixin,
    prepare_observation_for_snvla_inference,
)


class FakeRuntime(SNVLARuntimeMixin):
    def __init__(self) -> None:
        self._previous_narrations = ["first", "second"]
        self.latest_metrics = {
            "current_narration": "second",
            "previous_narrations": ["first"],
            "score": 0.75,
        }


def test_runtime_mixin_exposes_backend_independent_snapshot():
    runtime = FakeRuntime()

    output = runtime.get_snvla_output()

    assert isinstance(runtime, SNVLARuntime)
    assert output == SNVLAOutput(
        current_narration="second",
        previous_narrations=("first",),
        narration_history=("first", "second"),
        metrics={
            "current_narration": "second",
            "previous_narrations": ["first"],
            "score": 0.75,
        },
    )


def test_runtime_snapshot_does_not_alias_top_level_policy_state():
    runtime = FakeRuntime()

    output = runtime.get_snvla_output()
    runtime._previous_narrations.append("third")
    runtime.latest_metrics["score"] = 0.5

    assert output.narration_history == ("first", "second")
    assert output.metrics["score"] == 0.75


def test_inference_context_injects_json_history_without_mutating_observation():
    runtime = FakeRuntime()
    observation = {"observation.state": [1.0]}

    prepared = prepare_observation_for_snvla_inference(observation, runtime)

    assert json.loads(prepared["previous_narrations"]) == ["first", "second"]
    assert "previous_narrations" not in observation
