import torch
from src.model import TinyGPT


def sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample one token from model output probabilities."""
    logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        min_value = values[:, [-1]]
        logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
    cfg = checkpoint["config"]

    stoi = checkpoint["tokenizer_stoi"]
    itos = checkpoint["tokenizer_itos"]

    class LoadedTokenizer:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos

        def encode(self, s):
            return [self.stoi[c] for c in s if c in self.stoi]
        def decode(self, ids):
            return "".join(self.itos[i] for i in ids)

    tokenizer = LoadedTokenizer(stoi, itos)

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

    prompt = "ROMEO: "
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)

    for _ in range(100):
        x_cond = x[:, -cfg["model"]["max_seq_len"]:]
        logits, _ = model(x_cond)
        next_token = sample_next_token(logits[:, -1, :], temperature=0.8, top_k=20)
        x = torch.cat([x, next_token], dim=1)
    print(tokenizer.decode(x[0].tolist()))


if __name__ == "__main__":
    main()