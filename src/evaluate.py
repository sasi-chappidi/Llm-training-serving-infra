import math
import torch
from torch.utils.data import DataLoader

from src.model import TinyGPT
from src.dataset import NextTokenDataset


def main():
    checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
    cfg = checkpoint["config"]

    with open(cfg["data"]["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    stoi = checkpoint["tokenizer_stoi"]
    token_ids = [stoi[c] for c in text if c in stoi]

    split_idx = int(len(token_ids) * cfg["data"]["train_split"])
    val_ids = token_ids[split_idx:]

    val_ds = NextTokenDataset(val_ids, cfg["data"]["seq_len"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

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
    
    total_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    ppl = math.exp(avg_loss)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
    
    