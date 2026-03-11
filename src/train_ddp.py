import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.config import load_yaml
from src.utils import set_seed, ensure_dir
from src.tokenizer import CharTokenizer
from src.dataset import NextTokenDataset
from src.model import TinyGPT


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    """Close distributed training cleanly."""
    dist.destroy_process_group()


def main():
    cfg = load_yaml("configs/train.yaml")
    set_seed(cfg["seed"])
    rank, world_size = setup_ddp()

    with open(cfg["data"]["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    split_idx = int(len(token_ids) * cfg["data"]["train_split"])
    train_ids = token_ids[:split_idx]

    train_ds = NextTokenDataset(train_ids, cfg["data"]["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler)

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
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    ensure_dir(cfg["train"]["save_dir"])

    for epoch in range(cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        model.train()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                print(f"[rank {rank}] epoch={epoch} step={step} loss={loss.item():.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(cfg["train"]["save_dir"], "ddp_final.pt"))
        print("Saved DDP checkpoint.")

    cleanup_ddp()


if __name__ == "__main__":
    main()