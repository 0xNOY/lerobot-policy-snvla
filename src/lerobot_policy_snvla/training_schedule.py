from numbers import Integral

import numpy as np
import torch

_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MIX2 = np.uint64(0x94D049BB133111EB)


def stable_unit_phases(frame_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """Map stable integer frame IDs to deterministic values in the half-open unit interval."""
    frame_ids = torch.as_tensor(frame_ids)
    if frame_ids.is_floating_point() or frame_ids.is_complex():
        raise TypeError("frame_ids must contain integers")

    ids = frame_ids.detach().cpu().numpy().astype(np.uint64, copy=False)
    seed_bits = np.uint64(seed & _UINT64_MASK)
    with np.errstate(over="ignore"):
        mixed = ids + seed_bits + _SPLITMIX64_GAMMA
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _SPLITMIX64_MIX1
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _SPLITMIX64_MIX2
        mixed ^= mixed >> np.uint64(31)

    phases = ((mixed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).copy()
    return torch.from_numpy(phases).to(device=frame_ids.device)


def state_dropout_mask(
    frame_ids: torch.Tensor,
    epoch: int,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    """Select deterministic language-state dropout rows for one integer training epoch."""
    frame_ids = torch.as_tensor(frame_ids)
    if isinstance(epoch, bool) or not isinstance(epoch, Integral):
        raise TypeError("epoch must be an integer scalar")
    if not 0.0 <= ratio <= 0.5:
        raise ValueError("ratio must be between 0.0 and 0.5")
    if epoch <= 0 or ratio == 0.0:
        return torch.zeros_like(frame_ids, dtype=torch.bool)

    phase = stable_unit_phases(frame_ids, seed)
    previous = torch.floor((epoch - 1) * ratio + phase)
    current = torch.floor(epoch * ratio + phase)
    return current > previous
