import torch
from models.acu_net import AttentionGate


def test_attention_gate():
    gate = AttentionGate(
        encoder_channels=64,
        decoder_channels=128,
        inter_channels=32
    )

    encoder = torch.randn(2, 64, 32, 32, requires_grad=True)
    decoder = torch.randn(2, 128, 32, 32, requires_grad=True)

    captured = {}

    def capture_attention(module, inputs, output):
        captured["attention_map"] = output

    hook = gate.sigmoid.register_forward_hook(capture_attention)

    output = gate(encoder, decoder)

    hook.remove()

    attention_map = captured["attention_map"]

    assert output.shape == encoder.shape, (
        f"Output shape mismatch: {output.shape}"
    )

    assert attention_map.shape == (2, 1, 32, 32), (
        f"Attention map shape mismatch: {attention_map.shape}"
    )

    assert torch.all(attention_map >= 0).item()
    assert torch.all(attention_map <= 1).item()

    loss = output.mean()
    loss.backward()

    assert encoder.grad is not None
    assert decoder.grad is not None
    assert gate.encoder_projection.weight.grad is not None
    assert gate.decoder_projection.weight.grad is not None

    print("Output shape:", tuple(output.shape))
    print("Attention map shape:", tuple(attention_map.shape))
    print(
        "Attention weight range:",
        attention_map.min().item(),
        "to",
        attention_map.max().item()
    )
    print("Encoder gradients: PASS")
    print("Decoder gradients: PASS")
    print("Query/key projection gradients: PASS")
    print("All Attention Gate tests PASSED!")


if __name__ == "__main__":
    test_attention_gate()
