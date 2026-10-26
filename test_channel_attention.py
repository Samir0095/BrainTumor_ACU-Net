
import torch
from models.acu_net import ChannelAttention


def test_channel_attention():
    model = ChannelAttention(channels=64, reduction=16)

    x = torch.randn(2, 64, 32, 32, requires_grad=True)

    captured = {}

    def capture_attention(module, inputs, output):
        captured["weights"] = output

    hook = model.sigmoid.register_forward_hook(capture_attention)

    output = model(x)

    hook.remove()

    weights = captured["weights"]

    assert output.shape == x.shape, (
        f"Output shape mismatch: {output.shape}"
    )

    assert weights.shape == (2, 64), (
        f"Attention weight shape mismatch: {weights.shape}"
    )

    assert torch.all(weights >= 0).item()
    assert torch.all(weights <= 1).item()

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert model.mlp[0].weight.grad is not None
    assert model.mlp[2].weight.grad is not None

    print("Output shape:", tuple(output.shape))
    print("Channel attention weight shape:", tuple(weights.shape))
    print(
        "Attention weight range:",
        weights.min().item(),
        "to",
        weights.max().item()
    )
    print("Input gradients: PASS")
    print("MLP gradients: PASS")
    print("All Channel Attention tests PASSED!")


if __name__ == "__main__":
    test_channel_attention()
