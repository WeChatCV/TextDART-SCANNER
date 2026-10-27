"""Client configuration for remote rational reward models."""

import asyncio


def initialize_rational_client(self, config, default_model, supported_aspects, model_name):
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            f"{model_name} requires the `openai` package. "
            "Install with: pip install openai"
        ) from e

    self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8000/v1")
    self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
    self.vlm_model = config.extra_kwargs.get("vlm_model", default_model)
    self.max_concurrent = int(config.extra_kwargs.get("max_concurrent", 8))
    self.max_retries = int(config.extra_kwargs.get("max_retries", 5))
    self.timeout = float(config.extra_kwargs.get("timeout", 180.0))
    self.temperature = float(config.extra_kwargs.get("temperature", 0.1))
    self.max_tokens = int(config.extra_kwargs.get("max_tokens", 2048))

    raw_aspects = config.extra_kwargs.get("aspects")
    if raw_aspects is None:
        self.aspects = supported_aspects
    else:
        if not isinstance(raw_aspects, (list, tuple)) or not raw_aspects:
            raise TypeError(
                f"expected non-empty list/tuple for aspects, got {type(raw_aspects).__name__}: "
                f"{raw_aspects!r}"
            )
        self.aspects = tuple(str(a) for a in raw_aspects)
    unknown = [a for a in self.aspects if a not in supported_aspects]
    if unknown:
        raise ValueError(
            f"unsupported aspect(s) {unknown!r}; allowed: {list(supported_aspects)}"
        )

    self.client = AsyncOpenAI(base_url=self.api_base_url, api_key=self.api_key)
    self.semaphore = asyncio.Semaphore(max(1, self.max_concurrent))
