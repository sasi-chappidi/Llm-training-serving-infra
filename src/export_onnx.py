import os
import torch
from src.model import TinyGPT


def wrapper(model, x):
    logits, _ = model(x)
    return logits


def main():
    checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
    cfg = checkpoint["config"]

    model = TinyGPT(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs("triton_repo/tiny_llm_onnx/1", exist_ok=True)

    dummy_input = torch.randint(
        0,
        cfg["model"]["vocab_size"],
        (1, cfg["model"]["max_seq_len"]),
        dtype=torch.long,
    )

    class LogitsOnly(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            logits, _ = self.base_model(x)
            return logits

    export_model = LogitsOnly(model)
    export_model.eval()

    torch.onnx.export(
        export_model,
        dummy_input,
        "triton_repo/tiny_llm_onnx/1/model.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Exported ONNX model to triton_repo/tiny_llm_onnx/1/model.onnx")


if __name__ == "__main__":
    main()