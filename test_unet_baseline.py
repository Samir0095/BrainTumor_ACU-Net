import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_baseline import UNet


# ==========================================
# Configuration
# ==========================================

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "baseline_best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Test Baseline U-Net
# ==========================================

def test_baseline_unet():

    print("\nBaseline U-Net Verification")
    print("=" * 40)

    # 1. Create model
    model = UNet().to(DEVICE)

    print("Model instantiated successfully.")

    # 2. Load checkpoint
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    # 3. Evaluation mode
    model.eval()

    # 4. Dummy input
    x = torch.randn(
        1, 4, 240, 240,
        device=DEVICE
    )

    # 5. Forward pass
    with torch.no_grad():
        output = model(x)

    # 6. Verify shapes
    assert x.shape == (1, 4, 240, 240)

    assert output.shape == (1, 1, 240, 240)

    assert torch.isfinite(output).all()

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("Output contains only finite values.")

    print("\nBASELINE U-NET TEST PASSED!")


if __name__ == "__main__":
    test_baseline_unet()