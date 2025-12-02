import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import Net

def test_model_forward():
    model = Net()
    x = torch.randn(8, 1, 28, 28)  # バッチサイズ8
    output = model(x)

    assert output.shape == (8, 10)
    assert torch.all(torch.isfinite(output))
