
import torch
from models.acu_net import DualAttention


def test_dual_attention():
    model = DualAttention(channels=64, reduction=16)

    x = torch.randn(2, 64, 32, 32, requires_grad=True)

    captured = {}

    def capture_spatial(module, inputs, output):
        captured["spatial"] = output

    def capture_channel(module, inputs, output):
        captured["channel"] = output

    spatial_hook = model.spatial_attention.register_forward_hook(
        capture_spatial
    )
    channel_hook = model.channel_attention.register_forward_hook(
        capture_channel
    )

    output = model(x)

    spatial_hook.remove()
    channel_hook.remove()

    # Both branches should produce outputs
    assert "spatial" in captured
    assert "channel" in captured

    assert captured["spatial"].shape == x.shape
    assert captured["channel"].shape == x.shape

    # The dual-attention output must be the sum of both branches
    expected = captured["spatial"] + captured["channel"]

    assert torch.allclose(output, expected), (
        "Dual Attention output does not match the branch sum"
    )

    assert output.shape == x.shape, (
        f"Output shape mismatch: {output.shape}"
    )

    # Check gradient flow through both branches
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    assert model.spatial_attention.conv.weight.grad is not None

    assert model.channel_attention.mlp[0].weight.grad is not None
    assert model.channel_attention.mlp[2].weight.grad is not None

    print("Output shape:", tuple(output.shape))
    print("Spatial branch: PASS")
    print("Channel branch: PASS")
    print("Branch combination: PASS")
    print("Input gradients: PASS")
    print("Spatial attention gradients: PASS")
    print("Channel attention gradients: PASS")
    print("All Dual Attention tests PASSED!")


if __name__ == "__main__":
    test_dual_attention()
