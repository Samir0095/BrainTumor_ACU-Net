
import torch
from models.acu_net import SpatialAttention


def test_spatial_attention():
    # 1. Create the module and a sample input
    model = SpatialAttention(kernel_size=7)

    x = torch.randn(2, 64, 32, 32, requires_grad=True)

    # 2. Capture the attention map produced by the sigmoid
    captured = {}

    def capture_attention(module, inputs, output):
        captured["attention_map"] = output

    hook = model.sigmoid.register_forward_hook(capture_attention)

    # 3. Forward pass
    output = model(x)

    hook.remove()

    attention_map = captured["attention_map"]

    # 4. Verify tensor shapes
    assert output.shape == x.shape, (
        f"Output shape mismatch: {output.shape}"
    )

    assert attention_map.shape == (2, 1, 32, 32), (
        f"Attention map shape mismatch: {attention_map.shape}"
    )

    # 5. Verify attention weights are between 0 and 1
    assert torch.all(attention_map >= 0).item()
    assert torch.all(attention_map <= 1).item()

    # 6. Verify gradients can flow through the module
    loss = output.mean()
    loss.backward()

    assert x.grad is not None, "Input gradients were not calculated"

    assert model.conv.weight.grad is not None, (
        "Convolution weight gradients were not calculated"
    )

    print("Output shape:", tuple(output.shape))
    print("Attention map shape:", tuple(attention_map.shape))
    print("Attention weight range:",
          attention_map.min().item(),
          "to",
          attention_map.max().item())
    print("Input gradients: PASS")
    print("Convolution gradients: PASS")
    print("All Spatial Attention tests PASSED!")


if __name__ == "__main__":
    test_spatial_attention()
