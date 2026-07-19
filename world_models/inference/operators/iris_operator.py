import torch
from .base import OperatorABC
from typing import Dict, Any
import numpy as np


class IrisOperator(OperatorABC):
    """Operator for Iris transformer model: formats sequences and embeddings."""

    def __init__(self, seq_length: int = 512, vocab_size: int = 32000):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process Iris inputs: token sequences and optional embeddings.

        Expected inputs: {'tokens': list of ints or tensor, 'embeddings': optional tensor}
        """
        processed = {}

        # Process tokens
        if "tokens" in inputs:
            tokens = inputs["tokens"]
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            # Pad or truncate to seq_length
            if tokens.size(1) < self.seq_length:
                padding = torch.zeros(
                    tokens.size(0), self.seq_length - tokens.size(1), dtype=torch.long
                )
                tokens = torch.cat([tokens, padding], dim=1)
            elif tokens.size(1) > self.seq_length:
                tokens = tokens[:, : self.seq_length]

            processed["input_ids"] = tokens

        # Process embeddings if provided
        if "embeddings" in inputs:
            embeddings = inputs["embeddings"]
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)
            processed["embeddings"] = embeddings

        # Attention mask
        if "input_ids" in processed:
            processed["attention_mask"] = (processed["input_ids"] != 0).long()

        return processed
