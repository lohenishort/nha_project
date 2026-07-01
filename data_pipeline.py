#!/usr/bin/env python3
"""
Data pipeline for real language modeling benchmarking on NHA.
Provides streaming datasets (C4 / The Pile) tokenized with tiktoken.
"""

import random
import torch
from typing import Iterator, Tuple

# Predefined dummy texts for offline fallback when network is unavailable.
DUMMY_TEXTS = [
    "The Nearest Neighbor Attention (NHA) model is a recurrent neural network architecture designed for long-term memory tasks. Unlike standard transformers that attend to all past tokens, NHA maintains a set of gated memory slots that are updated dynamically at each step. This allows the model to scale to extremely long sequences with O(1) inference time per step.",
    "Artificial intelligence and machine learning have seen rapid progress in recent years. Large language models are trained on massive datasets like The Pile, C4, or Common Crawl to predict the next token. These models learn syntax, semantics, and world knowledge from billions of parameters.",
    "Rotary Position Embeddings, or RoPE, is a method to inject positional information into attention mechanisms. By rotating the query and key vectors in the complex plane, it models relative distances between tokens mathematically and naturally.",
    "This dataset is a fallback dummy corpus used because Hugging Face was not reachable. Perplexity benchmarking on this data should only be used to verify the code runs correctly end-to-end. In a real environment with network access, the actual C4 or Pile dataset will be streamed and used for evaluation.",
    "PyTorch provides flexible tensor operations and automatic differentiation which are essential for training deep neural networks. By vectorizing the recurrence relation, we can utilize GPUs efficiently and avoid slow Python loops.",
]

class LanguageDataLoader:
    """
    DataLoader that streams and tokenizes a dataset (e.g., C4 or The Pile)
    using Hugging Face datasets and tiktoken (GPT-2 tokenizer).
    Supports offline fallback to dummy texts if internet is unavailable.
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "gpt2",
        batch_size: int = 4,
        seq_len: int = 512
    ):
        """
        Initializes the LanguageDataLoader.

        Args:
            dataset_name: Name of the dataset to load (e.g., "c4", "EleutherAI/pile").
            tokenizer_name: Name of the tiktoken encoding to use (e.g., "gpt2").
            batch_size: Number of sequences per batch.
            seq_len: Length of each input sequence.
        """
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_token_id = 50256  # <|endoftext|> in GPT-2

        # Initialize tokenizer
        import tiktoken
        try:
            self.enc = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self.enc = tiktoken.get_encoding("gpt2")

        # Initialize dataset streaming
        self.offline = False
        from datasets import load_dataset
        try:
            if dataset_name == "c4" or dataset_name == "allenai/c4":
                # Use C4 english split
                self.dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            elif "pile" in dataset_name.lower():
                # Attempt general load, fallback if fails
                try:
                    self.dataset = load_dataset(dataset_name, split="train", streaming=True)
                except Exception:
                    self.dataset = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
            else:
                self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            print(f"Warning: Failed to load streaming dataset '{dataset_name}' (offline or connection issue: {e}).")
            print("Falling back to generating synthetic/dummy natural language text offline.")
            self.offline = True
            self.dataset = None

    def _yield_dummy_chunks(self) -> Iterator[list[int]]:
        """
        Yields tokenized chunks from dummy text.
        """
        while True:
            text = random.choice(DUMMY_TEXTS)
            tokens = self.enc.encode(text)
            if len(tokens) < 2:
                continue
            chunk_size = self.seq_len + 1
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                if len(chunk) < 2:
                    continue
                yield chunk

    def _get_tokenized_chunks(self) -> Iterator[list[int]]:
        """
        Streams documents and yields tokenized chunks up to seq_len + 1.
        Handles runtime exceptions during streaming and falls back to dummy text.
        """
        if self.offline:
            yield from self._yield_dummy_chunks()
        else:
            try:
                for example in self.dataset:
                    text = example.get("text", "")
                    if not text:
                        continue
                    tokens = self.enc.encode(text)
                    if len(tokens) < 2:
                        continue
                    chunk_size = self.seq_len + 1
                    for i in range(0, len(tokens), chunk_size):
                        chunk = tokens[i : i + chunk_size]
                        if len(chunk) < 2:
                            continue
                        yield chunk
            except Exception as e:
                print(f"Warning: Exception encountered while streaming from dataset '{self.dataset_name}' ({e}).")
                print("Falling back to generating synthetic/dummy natural language text offline.")
                self.offline = True
                yield from self._yield_dummy_chunks()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Yields batches of (input_ids, target_ids, attention_mask) tensors.
        """
        chunk_gen = self._get_tokenized_chunks()
        batch_inputs = []
        batch_targets = []
        batch_masks = []

        for chunk in chunk_gen:
            K = len(chunk)
            input_ids = chunk[:-1]
            target_ids = chunk[1:]
            M = K - 1

            if M < self.seq_len:
                pad_len = self.seq_len - M
                input_ids = input_ids + [self.pad_token_id] * pad_len
                target_ids = target_ids + [self.pad_token_id] * pad_len
                mask = [1.0] * M + [0.0] * pad_len
            else:
                mask = [1.0] * self.seq_len

            batch_inputs.append(input_ids)
            batch_targets.append(target_ids)
            batch_masks.append(mask)

            if len(batch_inputs) == self.batch_size:
                yield (
                    torch.tensor(batch_inputs, dtype=torch.long),
                    torch.tensor(batch_targets, dtype=torch.long),
                    torch.tensor(batch_masks, dtype=torch.float32)
                )
                batch_inputs = []
                batch_targets = []
                batch_masks = []
