from types import SimpleNamespace

import pytest
import torch

from lerobot_policy_snvla.modeling_molmoact2_snvla import (
    compile_training_flow_kernel,
    run_training_flow_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inductor_cudagraph_kernel_preserves_gradients_across_static_buckets():
    config = SimpleNamespace(
        compile_model=True,
        compile_backend="inductor",
        compile_mode="default",
        compile_cudagraphs=True,
        compile_fullgraph=True,
        compile_dynamic=False,
    )

    def objective(values):
        return (values.sin() * values).mean()

    compiled = compile_training_flow_kernel(objective, config)
    for length in (64, 96, 128):
        source = torch.linspace(-1.0, 1.0, length, device="cuda", dtype=torch.float32)
        eager_input = source.detach().clone().requires_grad_(True)
        eager_loss = objective(eager_input)
        eager_loss.backward()

        compiled_input = source.detach().clone().requires_grad_(True)
        compiled_loss = run_training_flow_kernel(
            compiled,
            config,
            prewarm_rope_cache=lambda: None,
            values=compiled_input,
        )
        compiled_loss.backward()

        torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=1e-5, atol=1e-6)
