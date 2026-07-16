import json
import signal
from pathlib import Path
from types import SimpleNamespace

from lerobot_policy_snvla.scripts import train_bf16_fsdp
from lerobot_policy_snvla.scripts.train_bf16_fsdp import (
    SignalCheckpointController,
    SignalCheckpointOptions,
    _signal_checkpoint_patches,
    parse_signal_checkpoint_options,
)


class FakeAccelerator:
    def __init__(self, *, is_main_process=True, remote_request=0):
        self.device = "cpu"
        self.is_main_process = is_main_process
        self.num_processes = 2
        self.process_index = 0 if is_main_process else 1
        self.remote_request = remote_request
        self.reduce_calls = 0

    def reduce(self, value, reduction="sum"):
        assert reduction == "max"
        self.reduce_calls += 1
        return value.new_tensor(max(int(value.item()), self.remote_request))


def make_controller(tmp_path: Path, *, remote_request=0, is_main_process=True):
    cfg = SimpleNamespace(
        output_dir=tmp_path,
        save_checkpoint=True,
        save_freq=100,
        steps=1_000,
    )
    accelerator = FakeAccelerator(
        is_main_process=is_main_process, remote_request=remote_request
    )
    controller = SignalCheckpointController(
        cfg, accelerator, signal.SIGUSR1, tmp_path / "signal_checkpoint_rank0.json"
    )
    return controller, accelerator, cfg


def test_parse_signal_checkpoint_options_is_disabled_by_default():
    options = parse_signal_checkpoint_options(["--batch_size=8", "--epochs=16"])

    assert options.signal_name is None
    assert options.pid_file is None
    assert options.remaining_argv == ["--batch_size=8", "--epochs=16"]


def test_parse_signal_checkpoint_options_consumes_explicit_disabled():
    options = parse_signal_checkpoint_options(
        ["--checkpoint-on-signal=disabled", "--batch_size=8"]
    )

    assert options.signal_name is None
    assert options.remaining_argv == ["--batch_size=8"]


def test_parse_signal_checkpoint_options_accepts_usr1_and_pid_file(tmp_path):
    pid_file = tmp_path / "worker.json"
    options = parse_signal_checkpoint_options(
        [
            "--checkpoint-on-signal=SIGUSR1",
            f"--signal-checkpoint-pid-file={pid_file}",
            "--batch_size=8",
        ]
    )

    assert options.signal_name == "SIGUSR1"
    assert options.pid_file == pid_file
    assert options.remaining_argv == ["--batch_size=8"]


def test_controller_coalesces_generations_observed_before_one_step(tmp_path):
    controller, accelerator, cfg = make_controller(tmp_path)
    controller.handle_signal(signal.SIGUSR1, None)
    controller.handle_signal(signal.SIGUSR1, None)

    assert controller.sync_after_update(7) is True
    assert accelerator.reduce_calls == 1
    assert cfg.save_freq == 7
    controller.restore_original_save_frequency()
    assert controller.sync_after_update(8) is False
    assert cfg.save_freq == 100


def test_controller_honors_a_request_received_by_another_rank(tmp_path):
    controller, accelerator, cfg = make_controller(tmp_path, remote_request=1)

    assert controller.sync_after_update(9) is True
    assert accelerator.reduce_calls == 1
    assert cfg.save_freq == 9


def test_controller_refreshes_epoch_resolved_save_frequency(tmp_path):
    controller, _, cfg = make_controller(tmp_path)
    cfg.save_freq = 250
    controller.refresh_original_save_frequency()
    controller.handle_signal(signal.SIGUSR1, None)

    assert controller.sync_after_update(7) is True
    controller.restore_original_save_frequency()
    assert cfg.save_freq == 250


def test_signal_save_restores_original_frequency_before_serializing(tmp_path, monkeypatch):
    controller, accelerator, cfg = make_controller(tmp_path)
    observed_save_frequencies = []

    def fake_save_checkpoint(*args, **kwargs):
        observed_save_frequencies.append(kwargs["cfg"].save_freq)

    monkeypatch.setattr(train_bf16_fsdp.lerobot_train, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(signal, "getsignal", lambda _signum: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda _signum, _handler: None)
    options = SignalCheckpointOptions("SIGUSR1", controller.pid_file, [])

    with _signal_checkpoint_patches(options, cfg, accelerator):
        active = train_bf16_fsdp._active_signal_checkpoint
        active.handle_signal(signal.SIGUSR1, None)
        assert active.sync_after_update(7) is True
        train_bf16_fsdp.lerobot_train.save_checkpoint(cfg=cfg)

    assert observed_save_frequencies == [100]
    assert cfg.save_freq == 100


def test_pid_metadata_is_atomic_and_cleaned_for_rank0(tmp_path):
    controller, _, _ = make_controller(tmp_path)
    path = controller.publish_pid_file()

    payload = json.loads(path.read_text())
    assert payload["pid"] > 0
    assert payload["process_index"] == 0
    assert payload["num_processes"] == 2
    assert payload["signal"] == "SIGUSR1"
    assert list(tmp_path.glob("*.tmp")) == []

    controller.remove_pid_file()
    assert not path.exists()


def test_non_main_rank_does_not_touch_pid_file(tmp_path):
    pid_file = tmp_path / "signal_checkpoint_rank0.json"
    pid_file.write_text('{"pid": 12345}')
    controller, _, _ = make_controller(tmp_path, is_main_process=False)

    assert controller.publish_pid_file() is None
    controller.remove_pid_file()
    assert json.loads(pid_file.read_text()) == {"pid": 12345}
