import torch
from fastapi import FastAPI
from serving.schemas import GenerateRequest
from src.model import TinyGPT

app = FastAPI(title="Tiny LLM API")

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


def sample_next_token(logits):
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    ids = tokenizer.encode(req.prompt)
    if not ids:
        return {"error": "Prompt contains unsupported characters."}

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    for _ in range(req.max_new_tokens):
        x_cond = x[:, -cfg["model"]["max_seq_len"] :]
        logits, _ = model(x_cond)
        next_token = sample_next_token(logits[:, -1, :])
        x = torch.cat([x, next_token], dim=1)
    return {"text": tokenizer.decode(x[0].tolist())}