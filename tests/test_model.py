import torch
from src.model import TinyGPT


def test_model_forward():
    model = TinyGPT(
        vocab_size=20,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.1,
    )

    x = torch.randint(0, 20, (2, 16))
    logits, loss = model(x, x)

    assert logits.shape == (2, 16, 20)
    assert loss is not None