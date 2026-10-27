# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/_http_clients.py
"""Shared HTTP client factories for reward-model HTTP/OpenAI servers.

Reward servers (vLLM, custom HTTP scorers, TextPecker, etc.) are
configured by URL in each reward's YAML block and are almost always
reachable directly from the trainer node over the LAN. Routing those
requests through whatever ``HTTP_PROXY`` / ``HTTPS_PROXY`` the user has
set up (e.g. a corporate proxy intended for *outbound* web traffic) is
a frequent source of 403 / "Attempt to access forbidden host" errors
that look like the reward server is broken when in fact it isn't.

To make rewards work out-of-the-box behind a corporate proxy:

- :func:`make_requests_session` builds a ``requests.Session`` with
  ``trust_env=False`` and ``proxies={}``.
- :func:`make_async_openai_client` builds an ``openai.AsyncOpenAI``
  with an ``httpx.AsyncClient(trust_env=False)`` transport.

Both default to **ignoring** env proxies. To opt back into env-driven
proxy resolution, set ``use_env_proxies: true`` in the reward's YAML
``extra_kwargs`` (or pass ``use_env_proxies=True`` programmatically).

Rewards that talk to a *pool* of interchangeable servers additionally use
:class:`ServerPool` / :class:`EndpointRotation`, which implement one shared
availability policy:

- **Startup**: unreachable servers are logged and dropped from the pool
  instead of aborting the run; only an entirely dead pool raises.
- **Runtime**: a failing server is put in a short cooldown and the request
  is immediately retried on the next healthy one; exponential backoff only
  kicks in once every server in the pool is cooling down.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

# httpx (used by openai.AsyncOpenAI) emits an INFO line for every HTTP
# request, which floods the training log with thousands of identical
# "HTTP Request: POST ... 200 OK" lines. Suppress at module import.
logging.getLogger("httpx").setLevel(logging.WARNING)


def make_requests_session(
    use_env_proxies: bool = False,
) -> requests.Session:
    """Build a ``requests.Session`` configured for reward-server traffic.

    Args:
        use_env_proxies: When ``False`` (default), the session ignores
            ``HTTP(S)_PROXY`` / ``NO_PROXY`` environment variables. When
            ``True``, behaves like a vanilla ``requests.Session``.

    Returns:
        A :class:`requests.Session` instance.
    """
    session = requests.Session()
    if not use_env_proxies:
        session.trust_env = False
        session.proxies = {}
    return session


def make_async_openai_client(
    *,
    base_url: str,
    api_key: str,
    timeout: Optional[float] = None,
    use_env_proxies: bool = False,
    **openai_kwargs: Any,
):
    """Build an ``openai.AsyncOpenAI`` client for reward-server traffic.

    When ``use_env_proxies=False`` (default), the underlying
    ``httpx.AsyncClient`` is constructed with ``trust_env=False`` so it
    ignores ``HTTP(S)_PROXY`` / ``NO_PROXY``. When ``True`` the default
    behavior of ``AsyncOpenAI`` (which inherits env proxies via
    ``httpx``) is preserved.

    Args:
        base_url: OpenAI-compatible base URL (must end with ``/v1`` if
            the server expects ``/v1/...`` paths; the textpecker /
            rational_rewards helpers handle that normalization upstream).
        api_key: API key string. vLLM servers typically accept the
            placeholder ``"EMPTY"``.
        timeout: Per-request timeout in seconds. Applied both to the
            ``httpx.AsyncClient`` transport and the ``AsyncOpenAI`` call.
        use_env_proxies: See module docstring.
        **openai_kwargs: Forwarded verbatim to ``AsyncOpenAI(...)``.

    Returns:
        An ``openai.AsyncOpenAI`` instance.

    Raises:
        ImportError: If ``openai`` is not installed. (``httpx`` ships as
            an ``openai`` dependency so it is always importable when
            ``openai`` is.)
    """
    from openai import AsyncOpenAI

    if use_env_proxies:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            **openai_kwargs,
        )

    import httpx

    httpx_kwargs: dict[str, Any] = {"trust_env": False}
    if timeout is not None:
        httpx_kwargs["timeout"] = timeout
    http_client = httpx.AsyncClient(**httpx_kwargs)
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        http_client=http_client,
        **openai_kwargs,
    )


def normalize_openai_base_url(server_url: str) -> str:
    """Ensure a server URL ends with the ``/v1`` prefix OpenAI clients expect."""
    url = server_url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def resolve_remote_model_name(
    session: requests.Session,
    base_url: str,
    override: Optional[str],
    timeout: float,
) -> str:
    """Return the served model id, auto-discovering via ``GET /v1/models``.

    When ``override`` is given we trust the caller. Otherwise we read the first
    entry of ``/models`` (vLLM always exposes it). Fails loudly when the server
    reports no served model, so we never silently send a wrong model name.
    """
    if override:
        return override
    resp = session.get(f"{base_url}/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if not data:
        raise RuntimeError(
            f"vLLM at {base_url} returned no served models. Start it with "
            f"--served-model-name or set an explicit model name in the reward config."
        )
    ids = [m.get("id") for m in data if m.get("id")]
    if not ids:
        raise RuntimeError(f"vLLM at {base_url} returned malformed /models payload: {data!r}")
    if len(ids) > 1:
        logger.warning(
            "vLLM at %s serves %d models %r; auto-picking %r. Pass an explicit "
            "model name to disambiguate.",
            base_url,
            len(ids),
            ids,
            ids[0],
        )
    return ids[0]


class EndpointRotation:
    """Round-robin index picker that temporarily skips endpoints that just failed.

    Failure marking is deliberately optimistic: a server is only sidelined for
    ``failure_cooldown`` seconds, and any later success clears the cooldown, so
    a one-off error never permanently shrinks the pool.
    """

    def __init__(
        self,
        num_endpoints: int,
        failure_cooldown: float = 60.0,
        start_index: int = 0,
    ):
        if num_endpoints <= 0:
            raise ValueError(f"expected at least one endpoint, got num_endpoints={num_endpoints}")
        if failure_cooldown < 0:
            raise ValueError(f"failure_cooldown must be >= 0, got {failure_cooldown}")
        self.num_endpoints = num_endpoints
        self.failure_cooldown = float(failure_cooldown)
        self._cooldown_until: List[float] = [0.0] * num_endpoints
        self._cycle = itertools.cycle(range(num_endpoints))
        self._lock = threading.Lock()
        for _ in range(start_index % num_endpoints):
            next(self._cycle)

    def next_index(self) -> int:
        """Next endpoint that is not cooling down (else the one recovering soonest)."""
        with self._lock:
            now = time.monotonic()
            for _ in range(self.num_endpoints):
                index = next(self._cycle)
                if self._cooldown_until[index] <= now:
                    return index
            return min(range(self.num_endpoints), key=self._cooldown_until.__getitem__)

    def has_available(self) -> bool:
        """True when at least one endpoint is outside its cooldown window."""
        with self._lock:
            now = time.monotonic()
            return any(until <= now for until in self._cooldown_until)

    def mark_failed(self, index: int) -> None:
        self._validate_index(index)
        with self._lock:
            self._cooldown_until[index] = time.monotonic() + self.failure_cooldown

    def mark_healthy(self, index: int) -> None:
        self._validate_index(index)
        with self._lock:
            self._cooldown_until[index] = 0.0

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < self.num_endpoints:
            raise IndexError(
                f"endpoint index {index} out of range for a pool of "
                f"{self.num_endpoints} endpoints"
            )


class ServerPool:
    """A round-robin pool of OpenAI-compatible clients for one server TYPE.

    ``httpx.AsyncClient`` (inside ``AsyncOpenAI``) binds to the event loop that
    first uses it. Async reward workers each have a thread-local loop
    (``_run_async_task``), so one process-wide client list is shared across
    loops and corrupts the transport: ``connection_lost`` on ``None``,
    ``Data should not be empty``, then a rank that never reaches the next
    NCCL collective. Clients are created lazily per thread instead.
    """

    def __init__(
        self,
        urls: Sequence[str],
        api_key: str,
        timeout: float,
        use_env_proxies: bool,
        model_name: Optional[str],
        session: requests.Session,
        probe_health: bool = True,
        failure_cooldown: float = 60.0,
        start_index: int = 0,
        pool_name: str = "reward server",
    ):
        base_urls = [normalize_openai_base_url(u) for u in urls]
        if not base_urls:
            raise ValueError(f"{pool_name} pool was given an empty URL list")
        if probe_health:
            base_urls = _keep_reachable_base_urls(session, base_urls, pool_name)
        self.base_urls = base_urls
        self.pool_name = pool_name
        self._client_api_key = api_key
        self._client_timeout = timeout
        self._client_use_env_proxies = use_env_proxies
        self._tls = threading.local()
        # All instances of a type are assumed to serve the same model; resolve
        # the name from the first instance that answers.
        self.model_name = self._discover_model_name(session, model_name, timeout=min(30.0, timeout))
        self._rotation = EndpointRotation(
            len(self.base_urls),
            failure_cooldown=failure_cooldown,
            start_index=start_index,
        )

    @property
    def clients(self) -> List[Any]:
        """This thread's AsyncOpenAI clients, one per ``base_urls`` entry."""
        return self._thread_clients()

    def _thread_clients(self) -> List[Any]:
        """Create or reuse the AsyncOpenAI list bound to this worker thread."""
        clients = getattr(self._tls, "clients", None)
        if clients is not None:
            return clients
        clients = [
            make_async_openai_client(
                base_url=base_url,
                api_key=self._client_api_key,
                timeout=self._client_timeout,
                use_env_proxies=self._client_use_env_proxies,
            )
            for base_url in self.base_urls
        ]
        self._tls.clients = clients
        return clients

    def next_client(self) -> Any:
        """Next client to use, skipping servers that are in failure cooldown."""
        return self._thread_clients()[self._rotation.next_index()]

    def has_available(self) -> bool:
        return self._rotation.has_available()

    def mark_failed(self, client: Any) -> None:
        self._rotation.mark_failed(self._index_of(client))

    def mark_healthy(self, client: Any) -> None:
        self._rotation.mark_healthy(self._index_of(client))

    def url_for(self, client: Any) -> str:
        return self.base_urls[self._index_of(client)]

    async def chat_with_failover(
        self,
        messages: List[Dict[str, Any]],
        *,
        retries: int,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        log_prefix: str = "remote chat",
    ) -> str:
        """Run one chat completion, moving to another server on every failure.

        Args:
            messages: OpenAI chat messages payload.
            retries: Total number of attempts across the whole pool.
            temperature: Sampling temperature.
            max_tokens: Decode budget.
            repetition_penalty: vLLM ``extra_body`` repetition penalty.
            log_prefix: Prefix identifying the caller in warning logs.

        Returns:
            The assistant message content (possibly an empty string).

        Raises:
            ValueError: If ``retries`` is not positive.
            RuntimeError: If every attempt failed; the last error is chained.
        """
        if retries <= 0:
            raise ValueError(f"expected retries >= 1 for {log_prefix}, got {retries}")
        last_error: Optional[Exception] = None
        client = self.next_client()
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"repetition_penalty": repetition_penalty},
                )
            except Exception as exc:  # pylint: disable=broad-except
                # Keep failover available for errors from any supported API client.
                last_error = exc
                failed_url = self.url_for(client)
                self.mark_failed(client)
                if attempt == retries - 1:
                    break
                client = self.next_client()
                # Another server is up -> switch over right away; only wait when
                # the whole pool is cooling down.
                wait = 0.0 if self.has_available() else min(5 * (2**attempt), 30)
                logger.warning(
                    "%s attempt %s/%s on %s failed: %s; retrying on %s in %ss",
                    log_prefix,
                    attempt + 1,
                    retries,
                    failed_url,
                    exc,
                    self.url_for(client),
                    wait,
                )
                if wait:
                    await asyncio.sleep(wait)
                continue
            self.mark_healthy(client)
            return response.choices[0].message.content or ""
        raise RuntimeError(
            f"{log_prefix} failed after {retries} attempts across "
            f"{self.base_urls}: {last_error}"
        ) from last_error

    def _discover_model_name(
        self, session: requests.Session, override: Optional[str], timeout: float
    ) -> str:
        last_error: Optional[Exception] = None
        for base_url in self.base_urls:
            try:
                return resolve_remote_model_name(session, base_url, override, timeout=timeout)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "cannot read the served model name from %s %s: %s",
                    self.pool_name,
                    base_url,
                    exc,
                )
        raise RuntimeError(
            f"no {self.pool_name} among {self.base_urls} could report a served "
            f"model name: {last_error}"
        ) from last_error

    def _index_of(self, client: Any) -> int:
        for index, owned in enumerate(self._thread_clients()):
            if owned is client:
                return index
        raise ValueError(
            f"client with base_url={getattr(client, 'base_url', None)!r} does "
            f"not belong to the {self.pool_name} pool {self.base_urls}"
        )


def _keep_reachable_base_urls(
    session: requests.Session,
    base_urls: List[str],
    pool_name: str,
    timeout: float = 5.0,
) -> List[str]:
    """Drop unreachable servers from a pool, logging each one that is skipped.

    A dead server at startup is a routine operational event (one node in a
    six-node vLLM pool rebooting), so it must not abort a multi-hour training
    run. Only a fully unreachable pool is fatal.
    """
    reachable: List[str] = []
    for base_url in base_urls:
        try:
            resp = session.get(f"{base_url}/models", timeout=timeout)
        except requests.RequestException as exc:
            logger.warning(
                "%s %s is unreachable (%s); dropping it from the pool",
                pool_name,
                base_url,
                exc,
            )
            continue
        if not resp.ok:
            logger.warning(
                "%s %s answered /models with HTTP %s; dropping it from the pool",
                pool_name,
                base_url,
                resp.status_code,
            )
            continue
        reachable.append(base_url)
    if not reachable:
        raise RuntimeError(f"no reachable {pool_name} among {base_urls}")
    if len(reachable) < len(base_urls):
        logger.warning(
            "%s pool running with %s/%s servers: %s",
            pool_name,
            len(reachable),
            len(base_urls),
            reachable,
        )
    return reachable
