class CharTokenizer:
    """
    A very simple character-level tokenizer.

    Example:
    text = "hello"
    unique characters = ['e', 'h', 'l', 'o']

    Each character gets an integer ID.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        """Convert string into list of token IDs."""
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """Convert token IDs back into string."""
        return "".join(self.itos[i] for i in ids)