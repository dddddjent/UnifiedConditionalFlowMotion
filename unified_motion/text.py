"""Frozen T5 encoding, kept outside the trainable motion model."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel


class TextEncoder:
    def __init__(
        self, model_name: str, device: torch.device, max_tokens: int = 128,
        *, legacy: bool | None = None,
    ) -> None:
        self.device = device
        self.max_tokens = max_tokens
        kwargs = {} if legacy is None else {"legacy": legacy}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
        self.model.requires_grad_(False)
        self.cache: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        assert prompts
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self.cache))
        if missing:
            tokens = self.tokenizer(
                missing, max_length=self.max_tokens, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            ids, masks = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
            encoded = self.model(input_ids=ids, attention_mask=masks).last_hidden_state.float()
            for prompt, value, mask in zip(missing, encoded, masks.bool(), strict=True):
                self.cache[prompt] = (value.detach().cpu(), mask.cpu())
        result = [self.cache[prompt] for prompt in prompts]
        for prompt in prompts:
            self.cache.move_to_end(prompt)
        while len(self.cache) > 4096:
            self.cache.popitem(last=False)
        encoded, masks = (
            torch.stack([entry[index] for entry in result]).to(self.device)
            for index in (0, 1)
        )
        return encoded, masks
