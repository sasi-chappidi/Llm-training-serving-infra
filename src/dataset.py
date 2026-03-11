import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """
    Build training examples for next-token prediction.

    If seq_len = 4 and data is [1, 2, 3, 4, 5], then:
      x = [1, 2, 3, 4]
      y = [2, 3, 4, 5]
    """

    def __init__(self, token_ids, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx + self.seq_len]
        y = self.token_ids[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)