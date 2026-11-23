"""A deterministic fake judge, for tests and for trying things without spending anything."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re

from .base import Completion, ProviderError


def mock_quality(text: str) -> float:
    """Latent quality in [0, 1]: an explicit ``[[q=0.73]]`` marker if present, else a hash of the text."""
    marker = re.search(r"\[\[q=([0-9.]+)\]\]", text)
    if marker:
        return float(marker.group(1))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class MockProvider:
    """Picks a winner with Bradley-Terry odds from the latent qualities, plus an optional first-position bias.

    Each verdict is seeded by the prompt, so results do not depend on which worker makes the call or in
    which order concurrent calls finish.
    """

    name = "mock"

    def __init__(self, model: str = "mock", seed: int = 0, sharpness: float = 8.0, position_bias: float = 0.0,
                 delay_s: float = 0.002, fail_rate: float = 0.0):
        self.model = model
        self.seed = seed
        self.sharpness = sharpness
        self.position_bias = position_bias
        self.delay_s = delay_s
        self.fail_rate = fail_rate
        self.calls = 0
        self._fail_rng = random.Random(seed)  # failures vary between attempts; verdicts do not

    async def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        texts = re.findall(r"<text_[ab]>\n(.*?)\n</text_[ab]>", user, re.S)
        if len(texts) != 2:
            raise ProviderError("mock judge could not find two texts in the prompt", retryable=False)
        rng = random.Random("%s|%s" % (self.seed, hashlib.sha256(user.encode()).hexdigest()))
        await asyncio.sleep(self.delay_s)
        if self.fail_rate and self._fail_rng.random() < self.fail_rate:
            raise ProviderError("mock failure")
        qa, qb = mock_quality(texts[0]), mock_quality(texts[1])
        p_a = 1 / (1 + math.exp(-(self.sharpness * (qa - qb) + self.position_bias)))
        winner = "A" if rng.random() < p_a else "B"
        return Completion(
            text=json.dumps({"reasoning": "Mock judge: latent quality %.2f vs %.2f." % (qa, qb), "winner": winner}),
            prompt_tokens=len(user) // 4,
            completion_tokens=30,
            cost_usd=0.0,
            latency_ms=int(self.delay_s * 1000),
        )

    async def close(self) -> None:
        pass
