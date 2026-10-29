import torch
from models.acu_net import ACUNet


def main():
    model = ACUNet()
    model.eval()

    x = torch.randn(1, 4, 240, 240)

    with torch.no_grad():
        enc1 = model.encoder1(x)
        attended = model.attention1(enc1)

    print("Encoder output:", tuple(enc1.shape))
    print("Attended output:", tuple(attended.shape))

    assert enc1.shape == attended.shape
    assert attended.shape == (1, 16, 240, 240)

    model.train()
    model.zero_grad()

    x = torch.randn(1, 4, 240, 240, requires_grad=True)
    enc1 = model.encoder1(x)
    attended = model.attention1(enc1)
    attended.mean().backward()

    assert x.grad is not None
    assert model.attention1.spatial_attention.conv.weight.grad is not None
    assert model.attention1.channel_attention.mlp[0].weight.grad is not None

    print("Shape preservation: PASS")
    print("Spatial attention gradients: PASS")
    print("Channel attention gradients: PASS")
    print("Encoder 1 attention test PASSED!")


if __name__ == "__main__":
    main()