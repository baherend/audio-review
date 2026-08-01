#!/usr/bin/env python3
"""
Phase 5 Step 2F — Masked Mean Pooling Parity Tests
===================================================
Tests that verify the mathematical correctness of masked mean pooling
versus simple mean pooling, and audits the current production architecture.

Scope:
  - Masked mean pooling mathematics
  - Padding behavior differences
  - Production architecture audit

Out of scope:
  - Model inference
  - Checkpoint loading
  - Hugging Face backbone
  - Streamlit integration

Constraints:
  - Deterministic synthetic tensors only
  - No Elgeish backbone
  - No checkpoint loading
  - No GPU required
  - Fast execution
"""

import torch
import torch.nn as nn


# ── Pooling Functions ────────────────────────────────────────────────────────

def simple_mean_pooling(lstm_out: torch.Tensor) -> torch.Tensor:
    """
    Simple mean pooling across the sequence dimension.
    This is the current production implementation in engine.py.

    Args:
        lstm_out: Tensor of shape [B, T, H]

    Returns:
        Pooled tensor of shape [B, H]
    """
    return lstm_out.mean(dim=1)


def masked_mean_pooling(
    lstm_out: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked mean pooling that excludes padded positions.
    This is the approved training implementation.

    Args:
        lstm_out: Tensor of shape [B, T, H]
        attention_mask: Tensor of shape [B, T] with 1 for valid, 0 for padded

    Returns:
        Pooled tensor of shape [B, H]
    """
    # Expand mask to match feature dimension
    valid_mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]

    # Sum valid positions
    summed = (lstm_out * valid_mask).sum(dim=1)  # [B, H]

    # Count valid positions (clamp to avoid division by zero)
    num_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

    # Mean of valid positions only
    pooled = summed / num_valid  # [B, H]

    return pooled


# ── Test 1: No-padding equivalence ──────────────────────────────────────────

def test_no_padding_equivalence():
    """
    Verify that simple mean pooling and masked mean pooling produce
    identical output when every feature-vector position is valid.
    """
    batch_size = 2
    seq_len = 4
    hidden_size = 8

    # Create deterministic tensor
    torch.manual_seed(42)
    lstm_out = torch.randn(batch_size, seq_len, hidden_size)

    # All positions valid
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Compute both pooling methods
    simple_pooled = simple_mean_pooling(lstm_out)
    masked_pooled = masked_mean_pooling(lstm_out, attention_mask)

    # They should be identical
    assert torch.allclose(simple_pooled, masked_pooled, atol=1e-6), (
        f"No-padding equivalence failed:\n"
        f"  Simple shape: {simple_pooled.shape}\n"
        f"  Masked shape: {masked_pooled.shape}\n"
        f"  Max diff: {(simple_pooled - masked_pooled).abs().max().item()}"
    )


# ── Test 2: Padding difference ──────────────────────────────────────────────

def test_padding_difference():
    """
    Create deterministic synthetic lstm_out values and a feature-vector
    attention mask containing padded positions. Verify that:
    - simple mean pooling includes padded positions
    - masked mean pooling excludes padded positions
    - the outputs are different
    - the masked result matches a manually calculated expected tensor
    """
    # Create a single batch item with 4 positions
    # Position 0, 1: valid
    # Position 2, 3: padded (zeros in attention mask)
    lstm_out = torch.tensor([
        [
            [1.0, 2.0, 3.0],  # Position 0: valid
            [4.0, 5.0, 6.0],  # Position 1: valid
            [0.0, 0.0, 0.0],  # Position 2: padded (but has values)
            [7.0, 8.0, 9.0],  # Position 3: padded (has values that should be excluded)
        ]
    ])  # Shape: [1, 4, 3]

    attention_mask = torch.tensor([
        [1, 1, 0, 0]  # Positions 0,1 valid; 2,3 padded
    ])  # Shape: [1, 4]

    # Simple mean pooling: includes ALL positions
    simple_pooled = simple_mean_pooling(lstm_out)
    # Simple mean = mean([1,4,0,7], [2,5,0,8], [3,6,0,9])
    #             = mean([12/4], [15/4], [18/4])
    #             = [3.0, 3.75, 4.5]
    expected_simple = torch.tensor([[3.0, 3.75, 4.5]])

    # Masked mean pooling: excludes padded positions 2, 3
    masked_pooled = masked_mean_pooling(lstm_out, attention_mask)
    # Masked mean = mean([1,4], [2,5], [3,6])  (only positions 0,1)
    #             = [2.5, 3.5, 4.5]
    expected_masked = torch.tensor([[2.5, 3.5, 4.5]])

    # Verify simple mean includes padded positions
    assert torch.allclose(simple_pooled, expected_simple, atol=1e-6), (
        f"Simple mean pooling mismatch:\n"
        f"  Expected: {expected_simple}\n"
        f"  Got: {simple_pooled}"
    )

    # Verify masked mean excludes padded positions
    assert torch.allclose(masked_pooled, expected_masked, atol=1e-6), (
        f"Masked mean pooling mismatch:\n"
        f"  Expected: {expected_masked}\n"
        f"  Got: {masked_pooled}"
    )

    # Verify the outputs are different
    assert not torch.allclose(simple_pooled, masked_pooled, atol=1e-6), (
        f"Simple and masked pooling should differ when padding exists:\n"
        f"  Simple: {simple_pooled}\n"
        f"  Masked: {masked_pooled}"
    )


# ── Test 3: Fully valid attention mask ──────────────────────────────────────

def test_fully_valid_mask():
    """
    Verify that masked pooling equals regular mean pooling when
    the mask contains only valid positions.
    """
    batch_size = 3
    seq_len = 5
    hidden_size = 16

    torch.manual_seed(123)
    lstm_out = torch.randn(batch_size, seq_len, hidden_size)

    # All positions valid
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    simple_pooled = simple_mean_pooling(lstm_out)
    masked_pooled = masked_mean_pooling(lstm_out, attention_mask)

    assert torch.allclose(simple_pooled, masked_pooled, atol=1e-6), (
        f"Fully valid mask should produce identical results:\n"
        f"  Max diff: {(simple_pooled - masked_pooled).abs().max().item()}"
    )


# ── Test 4: Batch with different valid lengths ──────────────────────────────

def test_batch_different_valid_lengths():
    """
    Use at least two batch items with different valid sequence lengths.
    Verify each item is pooled using its own valid positions.
    """
    # Batch item 0: 3 valid positions
    # Batch item 1: 1 valid position
    lstm_out = torch.tensor([
        [
            [1.0, 1.0],  # Position 0: valid
            [2.0, 2.0],  # Position 1: valid
            [3.0, 3.0],  # Position 2: valid
            [0.0, 0.0],  # Position 3: padded
        ],
        [
            [10.0, 10.0],  # Position 0: valid
            [0.0, 0.0],    # Position 1: padded
            [0.0, 0.0],    # Position 2: padded
            [0.0, 0.0],    # Position 3: padded
        ],
    ])  # Shape: [2, 4, 2]

    attention_mask = torch.tensor([
        [1, 1, 1, 0],  # Item 0: 3 valid
        [1, 0, 0, 0],  # Item 1: 1 valid
    ])  # Shape: [2, 4]

    masked_pooled = masked_mean_pooling(lstm_out, attention_mask)

    # Item 0: mean([1,1], [2,2], [3,3]) = [2.0, 2.0]
    # Item 1: mean([10,10]) = [10.0, 10.0]
    expected = torch.tensor([
        [2.0, 2.0],
        [10.0, 10.0],
    ])

    assert torch.allclose(masked_pooled, expected, atol=1e-6), (
        f"Batch different lengths failed:\n"
        f"  Expected:\n{expected}\n"
        f"  Got:\n{masked_pooled}"
    )


# ── Test 5: Production architecture audit ────────────────────────────────────

def test_production_uses_masked_mean_pooling():
    """
    AUDIT TEST — Production Architecture Verification

    This test inspects the current Wav2Vec2BiLSTMForSER.forward()
    implementation to confirm it uses masked mean pooling instead
    of simple mean pooling.

    This test verifies:
    - Production no longer uses lstm_out.mean(dim=1) as the only pooling
    - Production uses _get_feature_vector_attention_mask
    - Production applies mask-weighted summation
    - Production uses a protected denominator (clamp)

    Status: REGRESSION TEST — must pass after pooling implementation.
    """
    from pathlib import Path

    # Read the engine source
    engine_path = Path(__file__).parent.parent.parent / "src" / "inference" / "engine.py"
    source_code = engine_path.read_text(encoding="utf-8")

    # Check for masked pooling patterns
    has_feature_mask = "_get_feature_vector_attention_mask" in source_code
    has_valid_mask = "valid_mask" in source_code
    has_unsqueeze = "unsqueeze(-1)" in source_code
    has_weighted_sum = "(lstm_out * valid_mask).sum(dim=1)" in source_code
    has_clamped_denominator = ".clamp(min=1.0)" in source_code

    # Verify all masked pooling components are present
    assert has_feature_mask, (
        "Production should use _get_feature_vector_attention_mask. "
        "If this fails, the feature mask derivation is missing."
    )

    assert has_valid_mask, (
        "Production should use valid_mask. "
        "If this fails, the mask expansion is missing."
    )

    assert has_unsqueeze, (
        "Production should use unsqueeze(-1) to expand mask. "
        "If this fails, the mask expansion is missing."
    )

    assert has_weighted_sum, (
        "Production should use (lstm_out * valid_mask).sum(dim=1). "
        "If this fails, the masked summation is missing."
    )

    assert has_clamped_denominator, (
        "Production should use .clamp(min=1.0) to prevent division by zero. "
        "If this fails, the denominator protection is missing."
    )


# ── Test 6: Masked pooling with single valid position ───────────────────────

def test_single_valid_position():
    """
    Verify masked pooling works correctly when only one position is valid.
    """
    lstm_out = torch.tensor([
        [
            [0.0, 0.0, 0.0],  # Position 0: padded
            [5.0, 10.0, 15.0],  # Position 1: valid
            [0.0, 0.0, 0.0],  # Position 2: padded
        ]
    ])  # Shape: [1, 3, 3]

    attention_mask = torch.tensor([
        [0, 1, 0]
    ])  # Shape: [1, 3]

    masked_pooled = masked_mean_pooling(lstm_out, attention_mask)

    # Should be exactly the single valid position
    expected = torch.tensor([[5.0, 10.0, 15.0]])

    assert torch.allclose(masked_pooled, expected, atol=1e-6), (
        f"Single valid position failed:\n"
        f"  Expected: {expected}\n"
        f"  Got: {masked_pooled}"
    )
