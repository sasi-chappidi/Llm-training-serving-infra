import os
import torch
from torch.utils.data import DataLoader

from src.config import load_yaml
from src.utils import set_seed, ensure_dir
from src.tokenizer import CharTokenizer
from src.dataset import NextTokenDataset
from src.model import TinyGPT


def evaluate(model, loader, device):
    """Evaluate average loss on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    cfg = load_yaml("configs/train.yaml")
    set_seed(cfg["seed"])
    with open(cfg["data"]["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    split_idx = int(len(token_ids) * cfg["data"]["train_split"])
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = NextTokenDataset(train_ids, cfg["data"]["seq_len"])
    val_ds = NextTokenDataset(val_ids, cfg["data"]["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    # Use tokenizer vocab size directly, so config stays simple.
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    )

    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ensure_dir(cfg["train"]["save_dir"])
    best_val_loss = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            if step % cfg["train"]["log_every"] == 0:
                print(f"epoch={epoch} step={step} train_loss={loss.item():.4f}")

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_stoi": tokenizer.stoi,
                    "tokenizer_itos": tokenizer.itos,
                    "config": {
                        **cfg,
                        "model": {
                            **cfg["model"],
                            "vocab_size": tokenizer.vocab_size,
                        },
                    },
                },
                save_path,
            )
            print(f"Saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()