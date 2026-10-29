
import torch
from models.acu_net import ACUNet


def main():
    model = ACUNet()
    model.eval()

    x = torch.randn(1, 4, 240, 240)

    with torch.no_grad():
        output = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    assert output.shape == (1, 1, 240, 240)

    # Check gradients through the integrated attention modules.
    model.train()
    model.zero_grad()

    x = torch.randn(1, 4, 240, 240, requires_grad=True)
    output = model(x)
    output.mean().backward()

    assert x.grad is not None

    for name in (
        "attention_gate1",
        "attention_gate2",
        "attention_gate3",
        "attention_gate4"
    ):
        gate = getattr(model, name)

        assert gate.attention.weight.grad is not None
        assert gate.attention.bias.grad is not None

        print(f"{name}: gradient PASS")

    assert (
        model.dual_attention.spatial_attention.conv.weight.grad
        is not None
    )
    assert (
        model.dual_attention.channel_attention.mlp[0].weight.grad
        is not None
    )

    print("Dual Attention gradients: PASS")
    print("ACUNet integration test PASSED!")


if __name__ == "__main__":
    main()
